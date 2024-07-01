#include "tensor.h"

void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result) {
    if (tensor1->size != tensor2->size) {
        fprintf(stderr, "Tensors must have the same size for addition\n");
        return;
    }

    for (int i = 0; i < tensor1->size; i++) {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result) {
    if (tensor1->size != tensor2->size) {
        fprintf(stderr, "Tensors must have the same size for subtraction\n");
        return;
    }

    for (int i = 0; i < tensor1->size; i++) {
        result->data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, Tensor* result) {
    if (tensor1->size != tensor2->size) {
        fprintf(stderr, "Tensors must have the same size for element-wise multiplication\n");
        return;
    }

    for (int i = 0; i < tensor1->size; i++) {
        result->data[i] = tensor1->data[i] * tensor2->data[i];
    }
}

void assign_tensor_cpu(const Tensor* tensor, Tensor* result) {
    if (tensor->size != result->size) {
        fprintf(stderr, "Tensors must have the same size for assignment\n");
        return;
    }

    for (int i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i];
    }
}
