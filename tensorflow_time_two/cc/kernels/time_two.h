// kernel_example.h
#ifndef KERNEL_TIME_TWO_H_
#define KERNEL_TIME_TWO_H_

namespace tensorflow {

namespace functor {

template <typename Device>
struct TimeTwoFunctor {
  void operator()(const Device& d, 
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
                  float* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <>
struct TimeTwoFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d, 
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
                  float* out);
};
#endif

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
