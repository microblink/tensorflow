/**
 *
 *  FastSession
 *
 *  Created on: 31/03/2016
 *  Author: milijas, MicroBLINK
 *
 */

#ifndef TENSORFLOW_FAST_SESSION_H_
#define TENSORFLOW_FAST_SESSION_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class FastSession {
 public:

  virtual Status Run(const Tensor& input_tensor, Tensor* output_tensor) = 0;

  virtual ~FastSession() {}
};

FastSession* NewFastSession(GraphDef& graph);

}  // end namespace tensorflow

#endif  // TENSORFLOW_FAST_SESSION_H_
