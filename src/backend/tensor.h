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

Tensor* create_tensor(const float* data, const int* shape, int ndim);
void free_tensor(Tensor* tensor);
float get_element(const Tensor* tensor, const int* indices);
#endif 