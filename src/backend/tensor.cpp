#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cpu.h"
#include "tensor.h"

Tensor* create_tensor(const float* data, const int* shape, int ndim){
    
    if (data == nullptr || shape == nullptr || ndim <= 0) {
        fprintf(stderr, "Invalid input to create_tensor\n");
        return nullptr;
    }

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if(tensor == nullptr){
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if(tensor->shape == nullptr){
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor);
        return nullptr;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    
    tensor->size = 1;
    for(int i = 0; i < ndim; i++){
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if(tensor->strides == nullptr){
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
        return nullptr;
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->device = nullptr;

    return tensor;
}

void free_tensor(Tensor* tensor) {
    if (tensor != nullptr) {
        free(tensor->data);
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
    }
}

float get_element(const Tensor* tensor, const int* indices){
    int index = 0;
    for(int i = 0; i < tensor->ndim; i++){
        if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
            fprintf(stderr, "Index out of bounds\n"); 
        }
        index += indices[i] * tensor->strides[i];
    }
    return tensor->data[index];
}

Tensor* add_tensor(const Tensor* tensor1, const Tensor* tensor2){
    if (tensor1->ndim != tensor2->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions (%d and %d) for addition\n", tensor1->ndim, tensor2->ndim);
        return nullptr;
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            fprintf(stderr, "Tensors must have the same shape (%d and %d) at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
            free(shape);
            return nullptr;
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        free(shape);
        return nullptr;
    }

    Tensor* result = create_tensor(result_data, shape, ndim);
    if (result == nullptr) {
        free(shape);
        free(result_data);
        return nullptr;
    }

    add_tensor_cpu(tensor1, tensor2, result);
    return result;
}

Tensor* sub_tensor(const Tensor* tensor1, const Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions (%d and %d) for subtraction\n", tensor1->ndim, tensor2->ndim);
        return nullptr;
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            fprintf(stderr, "Tensors must have the same shape (%d and %d) at index %d for subtraction\n", tensor1->shape[i], tensor2->shape[i], i);
            free(shape);
            return nullptr;
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        free(shape);
        return nullptr;
    }

    Tensor* result = create_tensor(result_data, shape, ndim);
    if (result == nullptr) {
        free(shape);
        free(result_data);
        return nullptr;
    }

    sub_tensor_cpu(tensor1, tensor2, result);
    return result;
}

Tensor* elementwise_mul_tensor(const Tensor* tensor1, const Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions (%d and %d) for element-wise multiplication\n", tensor1->ndim, tensor2->ndim);
        return nullptr;
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            fprintf(stderr, "Tensors must have the same shape (%d and %d) at index %d for element-wise multiplication\n", tensor1->shape[i], tensor2->shape[i], i);
            free(shape);
            return nullptr;
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        free(shape);
        return nullptr;
    }

    Tensor* result = create_tensor(result_data, shape, ndim);
    if (result == nullptr) {
        free(shape);
        free(result_data);
        return nullptr;
    }

    elementwise_mul_tensor_cpu(tensor1, tensor2, result);
    return result;
}

Tensor* assign_tensor(const Tensor* tensor) {
    int ndim = tensor->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    for (int i = 0; i < ndim; i++) {
        shape[i] = tensor->shape[i];
    }

    float* result_data = (float*)malloc(tensor->size * sizeof(float));
    if (result_data == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        free(shape);
        return nullptr;
    }

    Tensor* result = create_tensor(result_data, shape, ndim);
    if (result == nullptr) {
        free(shape);
        free(result_data);
        return nullptr;
    }

    assign_tensor_cpu(tensor, result);
    return result;
}

Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
    int* shape = (int*)malloc(new_ndim * sizeof(int));
    if (shape == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return nullptr;
    }

    for (int i = 0; i < new_ndim; i++) {
        shape[i] = new_shape[i];
    }

    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_size *= shape[i];
    }

    if (new_size != tensor->size) {
        fprintf(stderr, "Cannot reshape tensor. Total number of elements in new shape (%d) does not match the current size of the tensor (%d).\n", new_size, tensor->size);
        free(shape);
        return nullptr;
    }

    float* result_data = (float*)malloc(tensor->size * sizeof(float));
    if (result_data == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        free(shape);
        return nullptr;
    }
    Tensor* result = create_tensor(result_data, shape, new_ndim);
    

    Tensor* reshaped_tensor = create_tensor(result_data, shape, new_ndim);
    if (reshaped_tensor == nullptr) {
        free(shape);
        free(result_data);
        return nullptr;
    }
    assign_tensor_cpu(tensor, reshaped_tensor);

    return reshaped_tensor;
}
