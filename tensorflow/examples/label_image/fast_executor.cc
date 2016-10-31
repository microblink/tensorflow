/**
 *
 *  FastExecutor
 *
 *  Created on: 31/03/2016
 *  Author: milijas, MicroBLINK
 *
 */

#include "fast_executor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

FastExecutor::FastExecutor(GraphDef& graph_def) {

    FunctionLibraryDefinition fdefs( OpRegistry::Global(), graph_def.library() );

    // Add default attributes to all new nodes in the graph.
    Status s = AddDefaultAttrsToGraphDef(&graph_def, fdefs, 0);
    if (!s.ok()) {
        printf("Default attrs to graph failed: %s", s.error_message().c_str());
        exit(-1);
    }

    Graph* device_graph = new Graph(&fdefs);
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = false;
    s = ConvertGraphDefToGraph(device_opts, graph_def, device_graph);
    if (!s.ok()) {
        printf("Convert graph failed: %s", s.error_message().c_str());
        exit(-1);
    }

    graph_ = device_graph;

    SessionOptions options;
    ConfigProto config;
    config.set_log_device_placement(true);
    config.set_inter_op_parallelism_threads(1);
    config.set_intra_op_parallelism_threads(1);
    options.config = config;

    std::vector<Device*> devices;
    DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices);
    DeviceMgr* mgr = new DeviceMgr( devices );

    Device* device = devices.front();
    device->op_segment()->AddHold("fast_session");

    auto runner = [this](std::function<void()> c) { c(); };
    const auto& optimizer_opts = options.config.graph_options().optimizer_options();
//    auto lib = NewFunctionLibraryRuntime(device, runner, graph_def.version(), &fdefs, optimizer_opts);
    auto lib = NewFunctionLibraryRuntime( mgr, Env::Default(), device, graph_def.version(), &fdefs, optimizer_opts );
    auto opseg = device->op_segment();

    FastExecutorParams params;
    params.device = device;
    params.function_library = lib;

    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate("fast_session", ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [](OpKernel* kernel) {
      // Do nothing because 'kernel' is owned by opseg above.
    };

    CHECK(params.create_kernel != nullptr);
    CHECK(params.delete_kernel != nullptr);

    params_ = params;

    Initialize();

}

FastExecutor::~FastExecutor() {

    for (NodeItem& item : nodes_) {
      params_.delete_kernel(item.kernel);
    }
    delete graph_;

    delete params_.function_library;
    delete params_.device;
}

Status FastExecutor::Initialize() {

    Status s;
    const int num_nodes = graph_->num_node_ids();
    nodes_.resize(num_nodes);

    total_input_tensors_ = 0;
    total_output_tensors_ = 0;

    // Preprocess every node in the graph to create an instance of op
    // kernel for each node;
    for (const Node* n : graph_->nodes()) {

      const int id = n->id();
      NodeItem* item = &nodes_[id];
      item->node = n;

      item->input_start = total_input_tensors_;
      total_input_tensors_ += n->num_inputs();

      item->output_attr_start = total_output_tensors_;
      total_output_tensors_ += n->num_outputs();

      s = params_.create_kernel(n->def(), &item->kernel);
      if (!s.ok()) {
        s = AttachDef(s, n->def());
        LOG(ERROR) << "Executor failed to create kernel. " << s;
        break;
      }
      CHECK(item->kernel);

    }

    AllocatorAttributes h;
    h.set_on_host(true);
    for (int i = 0; i < total_output_tensors_; i++)  output_attrs_.push_back(h);

    return s;
}

Status FastExecutor::Run(const Tensor& input_tensor, Tensor* output_tensor) {

    Status s;
    std::vector<Entry> input_tensors(total_input_tensors_);

    AllocatorAttributes h;
    h.set_on_host(true);
    input_tensors[0].alloc_attr = h;
    input_tensors[0].val = input_tensor;
    input_tensors[0].has_value = true;

    for ( int i = 0; i < nodes_.size(); ++i ) {
        LOG( INFO ) << "Node " << i << ": " << nodes_[ i ].node->name();
    }

    for (int i = 2; i < nodes_.size(); i++) {

        if (nodes_[i].node->name() != "INPUT_X") {

            s = ProcessNode(nodes_[i], input_tensors, output_tensor, i == nodes_.size() - 1);
            if (!s.ok()) return s;
        }
    }

    return s;
}

Status FastExecutor::ProcessNode(const NodeItem& node_item, std::vector<Entry>& input_tensors, Tensor* output_tensor, bool return_output) {

    Status s;
    OpKernel* op_kernel = node_item.kernel;
    const Node* node = node_item.node;

    TensorValueVec inputs(node->num_inputs());
    AllocatorAttributeVec input_alloc_attrs(node->num_inputs());

    OpKernelContext::Params ctxParams;
    Device* device = params_.device;
    ctxParams.device = device;
    ctxParams.function_library = params_.function_library;
    ctxParams.inputs = &inputs;
    ctxParams.input_alloc_attrs = &input_alloc_attrs;

//    VLOG(1) << "Process node: " << node->id() << " " << SummarizeNodeDef(node->def());

    Entry* first_input = input_tensors.data() + node_item.input_start;
    EntryVector outputs(node->num_outputs());

    bool is_input_dead = false;
    s = PrepareInputs(node_item, first_input, &inputs,
                      &input_alloc_attrs, &is_input_dead);

    if (!s.ok()) {
        printf("Processing inputs failed: %s", s.error_message().c_str());
        exit(-1);
    }

    ctxParams.op_kernel = op_kernel;
    ctxParams.is_input_dead = is_input_dead;
    ctxParams.output_attr_array =
        gtl::vector_as_array(&output_attrs_) + node_item.output_attr_start;

    OpKernelContext ctx(&ctxParams);

//    auto start_contraction = std::chrono::high_resolution_clock::now();

    device->Compute(CHECK_NOTNULL(op_kernel), &ctx);

//    auto stop_contraction = std::chrono::high_resolution_clock::now();
//    std::cout << "Compute duration: "
//                << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_contraction - start_contraction).count()
//                << " nanoseconds\n\n" << std::endl;

    // Process outputs
    s = ProcessOutputs(node_item, &ctx, &outputs);
    if (!s.ok()) {
        printf("Processing outputs failed: %s", s.error_message().c_str());
        exit(-1);
    }

    // Clear inputs
    int num_inputs = node->num_inputs();
    for (int i = 0; i < num_inputs; ++i) {
      (first_input + i)->val = *kEmptyTensor;
    }

    // Propagate outputs
    PropagateOutputs(node, input_tensors, outputs);

    // Return output tensor
    if (return_output && output_tensor != NULL) {

        float* in_data = outputs.back().val.flat<float>().data();
        float* out_data = (*output_tensor).flat<float>().data();
        for (int j = 0; j < (int) output_tensor->NumElements(); j++) {
            out_data[j] = in_data[j];
        }
    }

    return s;
}

Status FastExecutor::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead) {

  const Node* node = item.node;
  *is_input_dead = false;

  bool is_merge = IsMerge(node);
  for (int i = 0; i < node->num_inputs(); ++i) {
    const bool expect_ref = IsRefType(node->input_type(i));
    Entry* entry = first_input + i;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node));
        inp->tensor = &entry->val;
        *is_input_dead = true;
      }
      continue;
    }
    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }
      inp->tensor = &entry->val;
    } else {
      if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
        return AttachDef(
            errors::FailedPrecondition("Attempting to use uninitialized value ",
                                       item.kernel->def().input(i)),
            item.kernel->def());
      }
      if (expect_ref) {
        inp->mutex_if_ref = entry->ref_mu;
        inp->tensor = entry->ref;
      } else {
        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          mutex_lock l(*(entry->ref_mu));
          entry->val = *entry->ref;
        }
        inp->tensor = &entry->val;
      }
    }
  }

  return Status::OK();
}

Status FastExecutor::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs) {
  const Node* node = item.node;

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    LOG(WARNING) << this << " Compute status: " << s;
    return s;
  }

  for (int i = 0; i < node->num_outputs(); ++i) {
    TensorValue val = ctx->release_output(i);
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  SummarizeNodeDef(node->def())));
      }
    } else {
      Entry* out = &((*outputs)[i]);
      out->has_value = true;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype = val->dtype();
      if (val.is_ref()) dtype = MakeRefType(dtype);
      if (dtype == node->output_type(i)) {
        if (val.is_ref()) {
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
        } else {
          out->val = *val.tensor;
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(node->output_type(i)),
                                  " for node ", SummarizeNodeDef(node->def())));
      }
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  return s;
}

void FastExecutor::PropagateOutputs(const Node* node, std::vector<Entry>& input_tensors, const EntryVector& outputs) {

    for (const Edge* e : node->out_edges()) {

      const Node* dst_node = e->dst();
      const int dst_id = dst_node->id();
      const int src_slot = e->src_output();

      bool dst_need_input = !e->IsControlEdge();
      if (dst_need_input) {
        const NodeItem& dst_item = nodes_[dst_id];
        const int dst_slot = e->dst_input();
        int dst_loc = dst_item.input_start + dst_slot;
        input_tensors[dst_loc] = outputs[src_slot];
      }
    }
}

FastSession* NewFastSession(GraphDef& graph) {
    return new FastExecutor(graph);
}

}
