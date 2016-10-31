/**
 *
 *  FastExecutor
 *
 *  Created on: 31/03/2016
 *  Author: milijas, MicroBLINK
 *
 */

#ifndef TENSORFLOW_COMMON_RUNTIME_FAST_EXECUTOR_H_
#define TENSORFLOW_COMMON_RUNTIME_FAST_EXECUTOR_H_

#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "fast_session.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

namespace tensorflow {

struct NodeItem {
  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // ExecutorImpl::output_attrs_[output_attr_start] is the 1st
  // positional attribute for the 0th output of this node.
  int output_attr_start = 0;
};

static const Tensor* const kEmptyTensor = new Tensor;

static inline bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

struct FastExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;
};

struct Entry {
  Tensor val = *kEmptyTensor;  // A tensor value.
  Tensor* ref = nullptr;       // A tensor reference.
  mutex* ref_mu = nullptr;     // mutex for *ref if ref is not nullptr.
  bool has_value = false;      // Whether the value exists

  // The attributes of the allocator that creates the tensor.
  AllocatorAttributes alloc_attr;
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;
typedef gtl::InlinedVector<Entry, 4> EntryVector;

class FastExecutor : public FastSession {

public:

    FastExecutor(GraphDef& graph);
    virtual ~FastExecutor();

    ::tensorflow::Status Run(const Tensor& input_tensor, Tensor* output_tensor) override;

private:

    FastExecutorParams params_;
    const Graph* graph_;
    std::vector<NodeItem> nodes_;
    int total_input_tensors_ = 0;
    int total_output_tensors_ = 0;
    std::vector<AllocatorAttributes> output_attrs_;

    Status Initialize();
    Status PrepareInputs(const NodeItem& item, Entry* first_input, TensorValueVec* inputs, AllocatorAttributeVec* input_alloc_attrs, bool* is_input_dead);
    Status SetAllocAttrs();
    Status InferAllocAttr(const Node* n, const Node* dst, const DeviceNameUtils::ParsedName& local_dev_name, AllocatorAttributes* attr);
    Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx, EntryVector* outputs);
    void PropagateOutputs(const Node* node, std::vector<Entry>& input_tensors, const EntryVector& outputs);
    Status ProcessNode(const NodeItem& node_item, std::vector<Entry>& input_tensors, Tensor* output_tensor, bool return_output);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FAST_EXECUTOR_H_
