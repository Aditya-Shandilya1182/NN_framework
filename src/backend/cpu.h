#ifndef CPU_H
#define CPU_H

#include "tensor.h"

    void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result);
    void assign_tensor_cpu(const Tensor* tensor, Tensor* result);
    void reshape_tensor_cpu(Tensor* tensor, int* new_shape, int new_ndim);

#endif 
