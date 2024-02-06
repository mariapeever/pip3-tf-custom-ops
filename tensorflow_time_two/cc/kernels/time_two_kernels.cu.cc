/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "time_two.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#define SIZE 1024

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <>
__global__ void TimeTwoCudaKernel(
  const float* input, 
  const float* filter, 
  const int* dilations,
  const int* strides,
  const float* init_inputs,
  const float* fov_shape,
  const int* input_shape,
  const int* filter_shape,
  const int* init_shape,
  const long* output_shape,
  const int* h, 
  const int* g, 
  const int* a, 
  const int* b, 
  const int* c, 
  const int* p, 
  const int* e, 
  const float* fy_half, 
  const float* fx_half, 
  const float* fy1_half, 
  const float* fy2_half, 
  const float* fx1_half, 
  const float* fx2_half, 
  const float* fov_x,
  const float* fov_y,
  float* output) {

  int i = blockIdx.x;
  int j = threadIdx.y; 
  int k = threadIdx.x;

  if(i < output_shape[0] && j < output_shape[1] && k < output_shape[2]) {

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

    int diff = false;

    if (i > 0) {
      for(int l = y1; l < y2; l++) {
        for(int m = x1; m < x2; m++) {
          for(int n = 0; n < init_shape[3]; n++) {
            int init_index = l * h + m * init_shape[3] + n;
            if(init_inputs[(i * g) + init_index] != init_inputs[((i-1) * g) + init_index]) {
              diff = true;
            } 
          }
        }
      }
    } 

    if(i == 0 || diff == true) {
      int f_y1 = in_j-fy1_half;
      int f_y2 = in_j+fy2_half;
      int f_x1 = in_k-fx1_half;
      int f_x2 = in_k+fx2_half;

      for(int f = 0; f < filter_shape[2]; f++) {
        float filter_out = 0;
        for(int l = f_y1; l < f_y2; l++) {
          for(int m = f_x1; m < f_x2; m++) {
            float avg = 0;
            for(int n = 0; n < input_shape[3]; n++) {
              avg += input[i*b + l*a + m*input_shape[3] + n];
            }
            filter_out += (static_cast<float>(avg)/input_shape[3]) * filter[(l-f_y1)*e + (m-f_x1)*filter_shape[2] + f];
          }
        }
        output[i*p + j*c + k*output_shape[3] + f] = (filter_out > 1) ? 1 : (filter_out < 0) ? 0 : filter_out;
      }
    } else {
      for(int f = 0; f < filter_shape[2]; f++) {
        output[i*p + j*c + k*output_shape[3] + f] = output[(i-1)*p + j*c + k*output_shape[3] + f];
      }
    }
  }
}

// Define the CUDA kernel.
template <>
__global__ void TimeTwoCudaOutKernel(float* output, int* output_len, float* out) {

  int i = threadIdx.x;

  if(i < output_len) {
    out[i] = output[i];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <>
struct TimeTwoFunctor<GPUDevice> {
  void operator()(
    const GPUDevice& d, 
    const float* input, 
    const float* filter, 
    const int* dilations,
    const int* strides,
    const float* init_inputs,
    const float* fov_shape,
    const int* input_shape,
    const int* filter_shape,
    const int* init_shape,
    const long* output_shape,
    const float* fov_gap, 
    float* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.

    long * output_len;
    cudaMallocManaged(&output_len, output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * sizeof(long));

    output_len = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];

    float * output;
    cudaMallocManaged(&output, output_len * sizeof(float));

    float * fov_gap;
    cudaMallocManaged(&fov_gap, 2 * sizeof(float));

    fov_gap = {(static_cast<float>(init_shape[1]) - dilations[0] - dilations[2]) / output_shape[1], 
               (static_cast<float>(init_shape[2]) - dilations[1] - dilations[3]) / output_shape[2]};

    int *h, *g, *a, *b, *c, *p, *e;

    cudaMallocManaged(&h, sizeof(int));
    cudaMallocManaged(&g, sizeof(int));
    cudaMallocManaged(&a, sizeof(int));
    cudaMallocManaged(&b, sizeof(int));
    cudaMallocManaged(&c, sizeof(int));
    cudaMallocManaged(&p, sizeof(int));
    cudaMallocManaged(&e, sizeof(int));

    h = init_shape[2] * init_shape[3];
    g = init_shape[1] * h;

    a = input_shape[2] * input_shape[3];
    b = input_shape[1] * a;
    c = output_shape[2] * output_shape[3];
    p = output_shape[1] * c;

    e = filter_shape[1] * filter_shape[2];

    float *fy_half, *fx_half, *fy1_half, *fy2_half, *fx1_half, *fx2_half;

    cudaMallocManaged(&fy_half, sizeof(float));
    cudaMallocManaged(&fx_half, sizeof(float));
    cudaMallocManaged(&fy1_half, sizeof(float));
    cudaMallocManaged(&fy2_half, sizeof(float));
    cudaMallocManaged(&fx1_half, sizeof(float));
    cudaMallocManaged(&fx2_half, sizeof(float));

    fy_half = static_cast<float>(filter_shape[0])/2;
    fx_half = static_cast<float>(filter_shape[1])/2;

    fy1_half = floor(fy_half);
    fy2_half = ceil(fy_half);
    fx1_half = floor(fx_half);
    fx2_half = ceil(fx_half);

    float fov_x = static_cast<float>(fov_shape[0])/2;
    float fov_y = static_cast<float>(fov_shape[1])/2;

    int block_count = output_shape[0];

    dim3 threadsPerBlock(output_shape[1], output_shape[2]);

    TimeTwoCudaKernel
        <<<block_count, thread_per_block, 0, d.stream()>>>(
          input, 
          filter, 
          dilations, 
          strides, 
          init_inputs, 
          fov_shape, 
          input_shape, 
          filter_shape, 
          init_shape, 
          output_shape, 
          fov_gap, 
          h, 
          g, 
          a, 
          b, 
          c, 
          p, 
          e, 
          fy_half, 
          fx_half, 
          fy1_half, 
          fy2_half, 
          fx1_half, 
          fx2_half, 
          output);

    TimeTwoCudaOutKernel
        <<<1, output_len, 0, d.stream()>>>(output, output_len, out);

    cudaDeviceSynchronize();

    cudaFree(h);
    cudaFree(g);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(p);
    cudaFree(e);

    cudaFree(fy_half);
    cudaFree(fx_half);
    cudaFree(fy1_half);
    cudaFree(fy2_half);
    cudaFree(fx1_half);
    cudaFree(fx2_half);

    cudaFree(output_len);
    cudaFree(output);

  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TimeTwoFunctor<GPUDevice>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
