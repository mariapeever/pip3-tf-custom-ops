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

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor

    const Tensor& input_tensor = context->input(0);
    const Tensor& filter_tensor = context->input(1);
    const Tensor& dilations_tensor = context->input(2);
    const Tensor& strides_tensor = context->input(3);
    const Tensor& init_tensor = context->input(4);
    const Tensor& fov_tensor = context->input(5);

    auto input_flat = input_tensor.flat<float>();
    auto filter_flat = filter_tensor.flat<float>();
    auto dilations_flat = dilations_tensor.flat<float>();
    auto strides_flat = strides_tensor.flat<float>();
    auto init_flat = init_tensor.flat<float>();
    auto fov_shape_flat = fov_tensor.flat<float>();

    int input_shape [4];

    for(int i = 0; i < 4; i++) {
      input_shape[i] = input_tensor.dim_size(i);
    }

    int filter_shape [3];

    for(int i = 0; i < 3; i++) {
      filter_shape[i] = filter_tensor.dim_size(i);
    }

    int init_shape [4];

    for(int i = 0; i < 4; i++) {
      init_shape[i] = init_tensor.dim_size(i);
    }

    int input_len = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];

    float input [input_len];

    for(int i = 0; i < input_len; i++) {
      input[i] = input_flat(i);
    }

    int dilations [4] = {
      static_cast<int>(dilations_flat(0)),
      static_cast<int>(dilations_flat(1)),
      static_cast<int>(dilations_flat(2)),
      static_cast<int>(dilations_flat(3))
    };

    int strides [2] = {
      static_cast<int>(strides_flat(0)), 
      static_cast<int>(strides_flat(1))
    };

    int filter_len = filter_shape[0] * filter_shape[1] * filter_shape[2];

    float filter [filter_len];

    for(int i = 0; i < filter_len; i++) {
      filter[i] = filter_flat(i);
    }

    int init_len = init_shape[0] * init_shape[1] * init_shape[2] * init_shape[3];

    float init_inputs [init_len];

    for(int i = 0; i < init_len; i++) {
      init_inputs[i] = init_flat(i);
    }

    int fov_shape [2] = {
      static_cast<int>(fov_shape_flat(0)), 
      static_cast<int>(fov_shape_flat(1))
    };

    TensorShape out_shape = TensorShape({
        input_shape[0], 
        input_shape[1] - 2, 
        input_shape[2] - 2, 
        input_shape[3] * filter_shape[2]}); 
    
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, 
      out_shape,
      &output_tensor));

    auto output_flat = output_tensor->flat<float>();

    long output_shape [4] = {
      out_shape.dim_size(0), 
      out_shape.dim_size(1), 
      out_shape.dim_size(2), 
      out_shape.dim_size(3)
    };

    long output_len = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];

    float * output;
    output = new float[output_len];

    float fov_gap [2] = {(static_cast<float>(init_shape[1]) - dilations[0] - dilations[2]) / output_shape[1], 
                         (static_cast<float>(init_shape[2]) - dilations[1] - dilations[3]) / output_shape[2]};

    int g = init_shape[1] * init_shape[2];

    int a = input_shape[2] * input_shape[3];
    int b = input_shape[1] * a;
    int c = output_shape[2] * output_shape[3];
    int d = output_shape[1] * c;

    int e = filter_shape[1] * filter_shape[2];

    float fy_half = static_cast<float>(filter_shape[0])/2;
    float fx_half = static_cast<float>(filter_shape[1])/2;

    float fy1_half = floor(fy_half);
    float fy2_half = ceil(fy_half);
    float fx1_half = floor(fx_half);
    float fx2_half = ceil(fx_half);

    for (int i = 0; i < output_shape[0]; i++) {
      for (int j = 0; j < output_shape[1]; j++) {
        for (int k = 0; k < output_shape[2]; k++) {

          int in_j = j+dilations[0];
          int in_k = k+dilations[1];

          float t [2] = {(k+1) * fov_gap[1], (j+1) * fov_gap[0]}; // x y

          int s [2] = {static_cast<int>(floor(t[0] - fov_gap[1])), static_cast<int>(floor(t[1] - fov_gap[0]))}; // x y
          
          int xs = s[0]+dilations[1];
          int ys = s[1]+dilations[0];
          
          float fov_x = static_cast<float>(fov_shape[0])/2;
          float fov_y = static_cast<float>(fov_shape[1])/2;
          
          int y1 = ys - floor(fov_y);
          int y2 = ys + ceil(fov_y);
          int x1 = xs - floor(fov_x);
          int x2 = xs + ceil(fov_x);

          bool diff = false;

          if (i > 0) {
            for(int l = y1; l < y2; l++) {
              for(int m = x1; m < x2; m++) {
                int init_index = l * init_shape[2] + m;
                if(init_inputs[(i * g) + init_index] != init_inputs[((i-1) * g) + init_index]) {
                  diff = true;
                }
              }
            }
          } 

          if(i == 0 || diff == true) {
            int f_y1 = in_j-fy1_half;
            int f_y2 = in_j+fy2_half;
            int f_x1 = in_k-fx1_half;
            int f_x2 = in_k+fx2_half;

            for(int n = 0; n < input_shape[3]; n++) {
              for(int f = 0; f < filter_shape[2]; f++) {
                float filter_out = 0;
                for(int l = f_y1; l < f_y2; l++) {
                  for(int m = f_x1; m < f_x2; m++) {
                    filter_out += input[i*b + l*a + m*(filter_shape[2]*input_shape[3]) + f*input_shape[3] + n] * filter[(l-f_y1)*e + (m-f_x1)*filter_shape[2] + f];        
                  }
                }
                output[i*d + j*c + k*output_shape[3] + n*filter_shape[2] + f] = (filter_out > 1) ? 1 : (filter_out < 0) ? 0 : filter_out;
              }
            }
          } else {
            for(int o3 = 0; o3 < output_shape[3]; o3++) {
              output[i*d + j*c + k*output_shape[3] + o3] = output[(i-1)*d + j*c + k*output_shape[3] + o3];
            }
          }
        }
      }
    }

    for(int i = 0; i < output_len; i++) {
      output_flat(i) = output[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
