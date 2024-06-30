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