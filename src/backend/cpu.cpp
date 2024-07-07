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

void assign_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = tensor->data[i];
  }
}

void ones_like_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = 1.0;
  }
}

void zeros_like_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = 0.0;
  }
}

void transpose_1D_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->shape[0]; i++) {
    result_data[i] = tensor->data[i];
  }
}

void transpose_2D_tensor_cpu(Tensor* tensor, float* result_data) {
  int rows = tensor->shape[0];
  int cols = tensor->shape[1];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result_data[j * rows + i] = tensor->data[i * cols + j];
    }
  }
}

void transpose_3D_tensor_cpu(Tensor* tensor, float* result_data) {
  int batch = tensor->shape[0];
  int rows = tensor->shape[1];
  int cols = tensor->shape[2];

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < rows; j++) {
      for (int k = 0; k < cols; k++) {
        result_data[k * rows * batch + j * batch + i] = tensor->data[i * rows * cols + j * cols + k];
      }
    }
  }
}

void scalar_pow_tensor_cpu(float base, Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = powf(base, tensor->data[i]);
  }
}

void sigmoid_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    // avoid overflow
    if (tensor->data[i] >= 0) {
      float z = expf(-tensor->data[i]);
      result_data[i] = 1 / (1 + z);

    } else {
      float z = expf(tensor->data[i]);
      result_data[i] = z / (1 + z);
    }
  }
}

void sum_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
  if (axis == -1) {
    // Sum over all elements
    float sum = 0.0;
    for (int i = 0; i < tensor->size; i++) {
      sum += tensor->data[i];
    }
    *result_data = sum;
  } else {
    if (axis < 0 || axis >= tensor->ndim) {
      printf("Invalid axis");
      return;
    }

    int axis_stride = tensor->strides[axis];

    for (int i = 0; i < tensor->shape[axis]; i++) {
      for (int j = 0; j < size; j++) {
        int index = 0;
        int remainder = j;
        for (int k = tensor->ndim - 2; k >= 0; k--) {
          index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];
          remainder /= result_shape[k];
        }
        result_data[j] += tensor->data[index + i * axis_stride];
      }
    }
  }
}

void tensor_pow_scalar_cpu(Tensor* tensor, float exponent, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = powf(tensor->data[i], exponent);
  }
}

void log_tensor_cpu(Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = logf(tensor->data[i]);
  }
}

void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = scalar * tensor->data[i];
  }
}

void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
  for (int i = 0; i < tensor1->shape[0]; i++) {
    for (int j = 0; j < tensor2->shape[1]; j++) {
      float sum = 0.0;
      for (int k = 0; k < tensor1->shape[1]; k++) {
        sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
      }
      result_data[i * tensor2->shape[1] + j] = sum;
    }
  }
}

void scalar_div_tensor_cpu(float scalar, Tensor* tensor, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = scalar / tensor->data[i];
  }
}

void tensor_div_scalar_cpu(Tensor* tensor, float scalar, float* result_data) {
  for (int i = 0; i < tensor->size; i++) {
    result_data[i] = tensor->data[i] / scalar;
  }
}

void tensor_div_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
  for (int i = 0; i < tensor1->size; i++) {
    result_data[i] = tensor1->data[i] / tensor2->data[i];
  }
}

void make_contiguous_tensor_cpu(Tensor* tensor, float* result_data, int* new_strides) {
  for (int i = 0; i < tensor->size; i++) {
    int index = 0;
    int offset = i;
    for (int j = 0; j < tensor->ndim; j++) {
      index += (offset / new_strides[j]) * tensor->strides[j];
      offset %= new_strides[j];
    }
    result_data[i] = tensor->data[index];
  }

  // Free old data and update tensor properties
  free(tensor->data);
  free(tensor->strides);
  tensor->data = result_data;
  tensor->strides = new_strides;
}
