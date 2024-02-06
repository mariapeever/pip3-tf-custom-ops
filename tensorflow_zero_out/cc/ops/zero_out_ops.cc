/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("input: float")
    .Input("filter: float")
    .Input("dilations: float")
    .Input("strides: float")
    .Input("init_inputs: float")
    .Input("fov_shape: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {  
      shape_inference::ShapeHandle y;
      shape_inference::ShapeHandle x;
      shape_inference::ShapeHandle cy;
      shape_inference::ShapeHandle cx;
      shape_inference::ShapeHandle d;
      shape_inference::ShapeHandle f;   
      shape_inference::ShapeHandle out; 

      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, 1, &out)); 
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, 2, &y)); 
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 2, 3, &x));
      TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 3, &f));
      TF_RETURN_IF_ERROR(c->Merge(out, y, &out));
      TF_RETURN_IF_ERROR(c->Merge(out, x, &out));
      TF_RETURN_IF_ERROR(c->Merge(out, f, &out));

      c->set_output(0, out);
      return Status::OK();
    });
