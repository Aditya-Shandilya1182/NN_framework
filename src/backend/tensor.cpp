#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "tensor.h"

Tensor* create_tensor(const float* data, const int* shape, int ndim) {
  if (data == NULL || shape == NULL || ndim <= 0) {
    fprintf(stderr, "Invalid input to create_tensor\n");
    return NULL;
  }

  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  tensor->ndim = ndim;
  tensor->shape = (int*)malloc(ndim * sizeof(int));
  if (tensor->shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(tensor);
    return NULL;
  }
  memcpy(tensor->shape, shape, ndim * sizeof(int));

  tensor->size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor->size *= shape[i];
  }

  tensor->strides = (int*)malloc(ndim * sizeof(int));
  if (tensor->strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(tensor->shape);
    free(tensor);
    return NULL;
  }
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    tensor->strides[i] = stride;
    stride *= shape[i];
  }

  tensor->data = (float*)malloc(tensor->size * sizeof(float));
  if (tensor->data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(tensor->strides);
    free(tensor->shape);
    free(tensor);
    return NULL;
  }
  memcpy(tensor->data, data, tensor->size * sizeof(float));

  tensor->device = NULL;

  return tensor;
}

void free_tensor(Tensor* tensor) {
  if (tensor != NULL) {
    free(tensor->data);
    free(tensor->strides);
    free(tensor->shape);
    free(tensor);
  }
}

float get_element(const Tensor* tensor, const int* indices) {
  int index = 0;
  for (int i = 0; i < tensor->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
      fprintf(stderr, "Index out of bounds\n");
      return -1;  // Return an invalid value or handle this case appropriately
    }
    index += indices[i] * tensor->strides[i];
  }
  return tensor->data[index];
}

Tensor* add_tensor(const Tensor* tensor1, const Tensor* tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions (%d and %d) for "
            "addition\n",
            tensor1->ndim, tensor2->ndim);
    return NULL;
  }

  int ndim = tensor1->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape (%d and %d) at index %d for "
              "addition\n",
              tensor1->shape[i], tensor2->shape[i], i);
      free(shape);
      return NULL;
    }
    shape[i] = tensor1->shape[i];
  }

  float* result_data = (float*)malloc(tensor1->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    return NULL;
  }

  Tensor* result = create_tensor(result_data, shape, ndim);
  if (result == NULL) {
    free(shape);
    free(result_data);
    return NULL;
  }

  add_tensor_cpu(tensor1, tensor2, result);
  return result;
}

Tensor* sub_tensor(const Tensor* tensor1, const Tensor* tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions (%d and %d) for "
            "subtraction\n",
            tensor1->ndim, tensor2->ndim);
    return NULL;
  }

  int ndim = tensor1->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape (%d and %d) at index %d for "
              "subtraction\n",
              tensor1->shape[i], tensor2->shape[i], i);
      free(shape);
      return NULL;
    }
    shape[i] = tensor1->shape[i];
  }

  float* result_data = (float*)malloc(tensor1->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    return NULL;
  }

  Tensor* result = create_tensor(result_data, shape, ndim);
  if (result == NULL) {
    free(shape);
    free(result_data);
    return NULL;
  }

  sub_tensor_cpu(tensor1, tensor2, result);
  return result;
}

Tensor* elementwise_mul_tensor(const Tensor* tensor1, const Tensor* tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions (%d and %d) for "
            "element-wise multiplication\n",
            tensor1->ndim, tensor2->ndim);
    return NULL;
  }

  int ndim = tensor1->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape (%d and %d) at index %d for "
              "element-wise multiplication\n",
              tensor1->shape[i], tensor2->shape[i], i);
      free(shape);
      return NULL;
    }
    shape[i] = tensor1->shape[i];
  }

  float* result_data = (float*)malloc(tensor1->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    return NULL;
  }

  Tensor* result = create_tensor(result_data, shape, ndim);
  if (result == NULL) {
    free(shape);
    free(result_data);
    return NULL;
  }

  elementwise_mul_tensor_cpu(tensor1, tensor2, result);
  return result;
}

Tensor* assign_tensor(const Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    return NULL;
  }

  Tensor* result = create_tensor(result_data, shape, ndim);
  if (result == NULL) {
    free(shape);
    free(result_data);
    return NULL;
  }

  assign_tensor_cpu(tensor, result);
  return result;
}

Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
  int* shape = (int*)malloc(new_ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < new_ndim; i++) {
    shape[i] = new_shape[i];
  }

  int new_size = 1;
  for (int i = 0; i < new_ndim; i++) {
    new_size *= shape[i];
  }

  if (new_size != tensor->size) {
    fprintf(stderr,
            "Cannot reshape tensor. Total number of elements in new shape (%d) "
            "does not match the current size of the tensor (%d).\n",
            new_size, tensor->size);
    free(shape);
    return NULL;
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    return NULL;
  }

  Tensor* reshaped_tensor = create_tensor(result_data, shape, new_ndim);
  if (reshaped_tensor == NULL) {
    free(shape);
    free(result_data);
    return NULL;
  }
  assign_tensor_cpu(tensor, reshaped_tensor);

  return reshaped_tensor;
}

Tensor* ones_like_tensor(Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  ones_like_tensor_cpu(tensor, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* zeros_like_tensor(Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  zeros_like_tensor_cpu(tensor, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* transpose_tensor(Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[ndim - 1 - i];
  }

  int size = tensor->size;

  float* result_data = (float*)malloc(size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  switch (ndim) {
    case 1:
      transpose_1D_tensor_cpu(tensor, result_data);
      break;
    case 2:
      transpose_2D_tensor_cpu(tensor, result_data);
      break;
    case 3:
      transpose_3D_tensor_cpu(tensor, result_data);
      break;
    default:
      fprintf(stderr, "Transpose only supports tensors up to 3 dimensions.\n");
      exit(-1);
  }
  return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_mul_tensor(Tensor* tensor, float scalar) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }
  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  scalar_mul_tensor_cpu(tensor, scalar, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
  // MxN @ NxP = MxP
  // Check if tensors have compatible shapes for matrix multiplication
  if (tensor1->shape[1] != tensor2->shape[0]) {
    fprintf(stderr,
            "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n",
            tensor1->shape[0], tensor1->shape[1], tensor2->shape[0],
            tensor2->shape[1]);
    exit(1);
  }

  int ndim = tensor1->ndim + tensor2->ndim - 2;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  for (int i = 0; i < tensor1->ndim - 1; i++) {
    shape[i] = tensor1->shape[i];
  }
  for (int i = tensor1->ndim - 1; i < ndim; i++) {
    shape[i] = tensor2->shape[i - tensor1->ndim + 2];
  }

  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }

  float* result_data = (float*)malloc(size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  matmul_tensor_cpu(tensor1, tensor2, result_data);
  return create_tensor(result_data, shape, ndim);
}
Tensor* tensor_pow_scalar(Tensor* tensor, float exponent) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }
  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor_pow_scalar_cpu(tensor, exponent, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* scalar_pow_tensor(float base, Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }
  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  scalar_pow_tensor_cpu(base, tensor, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* sigmoid_tensor(Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sigmoid_tensor_cpu(tensor, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim) {
  int ndim;
  int* shape;

  if (axis > tensor->ndim - 1) {
    fprintf(stderr,
            "Error: axis argument %d must be smaller than tensor dimension %d",
            axis, tensor->ndim);
  }

  if (axis == -1) {
    shape = (int*)malloc(sizeof(int));
    shape[0] = 1;
    ndim = 1;
  } else {
    shape = (int*)malloc((tensor->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < tensor->ndim; ++i) {
      if (i != axis) {
        shape[j++] = tensor->shape[i];
      }
    }
    ndim = tensor->ndim - 1;
  }

  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }

  float* result_data = (float*)calloc(axis_size, sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  sum_tensor_cpu(tensor, result_data, axis_size, shape, axis);

  if (keepdim) {
    if (axis == -1) {
      ndim = tensor->ndim;
      shape = (int*)malloc((tensor->ndim) * sizeof(int));
      for (int i = 0; i < tensor->ndim; i++) {
        shape[i] = 1;
      }
    } else {
      shape = (int*)malloc((tensor->ndim) * sizeof(int));
      for (int i = 0; i < tensor->ndim; i++) {
        shape[i] = tensor->shape[i];
      }
      shape[axis] = 1;
      ndim = tensor->ndim;
    }
  }
  return create_tensor(result_data, shape, ndim);
}

Tensor* tensor_div_scalar(Tensor* tensor, float scalar) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor_div_scalar_cpu(tensor, scalar, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions %d and %d for "
            "element-wise division\n",
            tensor1->ndim, tensor2->ndim);
    exit(1);
  }

  int ndim = tensor1->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape %d and %d at index %d for "
              "division\n",
              tensor1->shape[i], tensor2->shape[i], i);
      exit(1);
    }
    shape[i] = tensor1->shape[i];
  }

  float* result_data = (float*)malloc(tensor1->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor_div_tensor_cpu(tensor1, tensor2, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* log_tensor(Tensor* tensor) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  log_tensor_cpu(tensor, result_data);
  return create_tensor(result_data, shape, ndim);
}

Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2) {
  int ndim = tensor->ndim;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  shape[axis1] = tensor->shape[axis2];
  shape[axis2] = tensor->shape[axis1];

  int size = tensor->size;

  float* result_data = (float*)malloc(size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  assign_tensor_cpu(tensor, result_data);

  Tensor* new_tensor = create_tensor(result_data, shape, ndim);
  for (int i = 0; i < ndim; i++) {
    new_tensor->strides[i] = tensor->strides[i];
  }
  new_tensor->strides[axis1] = tensor->strides[axis2];
  new_tensor->strides[axis2] = tensor->strides[axis1];
  make_contiguous(new_tensor);
  return new_tensor;
}
void make_contiguous(Tensor* tensor) {
  int* new_strides = (int*)malloc(tensor->ndim * sizeof(int));
  if (new_strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
  }

  // Calculate new strides assuming C-contiguous order
  int stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; i--) {
    new_strides[i] = stride;
    stride *= tensor->shape[i];
  }

  float* result_data = (float*)malloc(tensor->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
  }
  make_contiguous_tensor_cpu(tensor, result_data, new_strides);
}
