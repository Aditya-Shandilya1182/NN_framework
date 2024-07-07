#ifndef CPU_H
#define CPU_H

#include "tensor.h"

    void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void assign_tensor_cpu(Tensor* tensor, float* result_data);
    void assign_tensor_cpu(const Tensor* tensor, Tensor* result);
    void reshape_tensor_cpu(Tensor* tensor, int* new_shape, int new_ndim);
    void ones_like_tensor_cpu(Tensor* tensor, float* result_data);
    void zeros_like_tensor_cpu(Tensor* tensor, float* result_data);
    void transpose_1D_tensor_cpu(Tensor* tensor, float* result_data);
    void transpose_2D_tensor_cpu(Tensor* tensor, float* result_data);
    void transpose_3D_tensor_cpu(Tensor* tensor, float* result_data);
    void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
    void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data);
    void log_tensor_cpu(Tensor* tensor, float* result_data);
    void tensor_pow_scalar_cpu(Tensor* tensor, float exponent, float* result_data);
    void scalar_pow_tensor_cpu(float base, Tensor* tensor, float* result_data);
    void sigmoid_tensor_cpu(Tensor* tensor, float* result_data);
    void sum_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis);
    void scalar_div_tensor_cpu(float scalar, Tensor* tensor, float* result_data);
    void tensor_div_scalar_cpu(Tensor* tensor, float scalar, float* result_data);
    void tensor_div_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
    void make_contiguous_tensor_cpu(Tensor* tensor, float* result_data, int* new_strides);
    
#endif 

//g++ -c cpu.cpp -o cpu.o
// g++ -c tensor.cpp -o tensor.o
// g++ -shared -o tensor_lib.so cpu.o tensor.o