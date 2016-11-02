/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

#ifdef TF_KERNEL_BENCHMARK
#include "tensorflow/examples/label_image/ElapsedTimer.hpp"
#endif

namespace tensorflow {

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, locality),
                  allocator),
      allocator_(allocator) {}

ThreadPoolDevice::~ThreadPoolDevice() {}

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
#ifdef TF_KERNEL_BENCHMARK
      ElapsedTimer timer;
#endif
    op_kernel->Compute(context);
#ifdef TF_KERNEL_BENCHMARK
    double tm = timer.toc();
    // first find pair in kernel_times_
    // we need to do linear search because we want to maintain the order of kernels as they were invoked
    // simply putting results into a map will change the ordering
    size_t index;
    for( index = 0; index < kernel_times_.size(); ++index ) {
        if( kernel_times_[ index ].first == op_kernel->name() ) break;
    }
    if ( index < kernel_times_.size() ) {
        kernel_times_[ index ].second += tm;
    } else {
        kernel_times_.emplace_back( op_kernel->name(), tm );
    }
#endif
  }
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
  }
  *tensor = parsed;
  return Status::OK();
}

}  // namespace tensorflow
