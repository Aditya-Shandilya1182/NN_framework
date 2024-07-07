#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
    char* device;
} Tensor;

extern "C" {
    Tensor* create_tensor(const float* data, const int* shape, int ndim);
    void free_tensor(Tensor* tensor);
    float get_element(const Tensor* tensor, const int* indices);
    Tensor* add_tensor(const Tensor* tensor1, const Tensor* tensor2);
    Tensor* sub_tensor(const Tensor* tensor1, const Tensor* tensor2);
    Tensor* elementwise_mul_tensor(const Tensor* tensor1, const Tensor* tensor2);
    Tensor* assign_tensor(const Tensor* tensor);
    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* ones_like_tensor(Tensor* tensor);
    Tensor* zeros_like_tensor(Tensor* tensor);
    Tensor* transpose_tensor(Tensor* tensor);
    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* scalar_mul_tensor(Tensor* tensor, float scalar);
    Tensor* scalar_pow_tensor(float base, Tensor* tensor);
    Tensor* tensor_pow_scalar(Tensor* tensor, float exponent);
    Tensor* sigmoid_tensor(Tensor* tensor);
    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim);
    Tensor* tensor_div_scalar(Tensor* tensor, float scalar);
    Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* log_tensor(Tensor* tensor);
    Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2);
    void make_contiguous(Tensor* tensor);
}

#endif
