///////////////////////////////////////////////////////////////////////
// File:        cudabackend.cpp
// Description: Optional CUDA helper for LSTM inference.
// Author:      GitHub Copilot
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
///////////////////////////////////////////////////////////////////////

#include "cudabackend.h"

#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(HAVE_CUDA)
#  include <cublas_v2.h>
#  include <cuda_runtime_api.h>
#endif

namespace tesseract {

#if defined(HAVE_CUDA) && defined(FAST_FLOAT)
namespace {

class CudaEnvironment {
public:
  static CudaEnvironment &Instance() {
    static CudaEnvironment instance;
    return instance;
  }

  bool Available(std::string *error) {
    if (!initialized_) {
      initialized_ = true;
      int device_count = 0;
      auto cuda_status = cudaGetDeviceCount(&device_count);
      if (cuda_status != cudaSuccess || device_count <= 0) {
        available_ = false;
        last_error_ = cuda_status == cudaSuccess ? "no CUDA device available" : cudaGetErrorString(cuda_status);
      } else if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
        available_ = false;
        last_error_ = "failed to create cuBLAS handle";
      } else {
        available_ = true;
        last_error_.clear();
      }
    }
    if (!available_ && error != nullptr) {
      *error = last_error_;
    }
    return available_;
  }

  cublasHandle_t handle() const {
    return handle_;
  }

private:
  CudaEnvironment() : initialized_(false), available_(false), handle_(nullptr) {}
  ~CudaEnvironment() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  bool initialized_;
  bool available_;
  cublasHandle_t handle_;
  std::string last_error_;
};

bool CheckCuda(cudaError_t status, std::string *error) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error != nullptr) {
    *error = cudaGetErrorString(status);
  }
  return false;
}

} // namespace
#endif

CudaMatrix::CudaMatrix() : rows_(0), cols_(0), device_weights_(nullptr), device_input_(nullptr), device_output_(nullptr) {}

CudaMatrix::~CudaMatrix() {
#if defined(HAVE_CUDA) && defined(FAST_FLOAT)
  if (device_weights_ != nullptr) {
    cudaFree(device_weights_);
  }
  if (device_input_ != nullptr) {
    cudaFree(device_input_);
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
  }
#endif
}

void CudaMatrix::Invalidate() {
  rows_ = 0;
  cols_ = 0;
}

bool CudaMatrix::Prepare(const GENERIC_2D_ARRAY<TFloat> &weights, std::string *error) {
  return EnsureWeights(weights, error);
}

bool CudaMatrix::EnsureWeights(const GENERIC_2D_ARRAY<TFloat> &weights, std::string *error) {
#if defined(HAVE_CUDA) && defined(FAST_FLOAT)
  if (!CudaEnvironment::Instance().Available(error)) {
    return false;
  }
  int rows = weights.dim1();
  int cols = weights.dim2() - 1;
  if (rows == rows_ && cols == cols_ && device_weights_ != nullptr) {
    return true;
  }
  rows_ = rows;
  cols_ = cols;
  std::vector<TFloat> packed(rows * cols);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(&packed[r * cols], weights[r], cols * sizeof(TFloat));
  }
  if (device_weights_ != nullptr) {
    cudaFree(device_weights_);
    device_weights_ = nullptr;
  }
  if (device_input_ != nullptr) {
    cudaFree(device_input_);
    device_input_ = nullptr;
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  if (!CheckCuda(cudaMalloc(&device_weights_, packed.size() * sizeof(TFloat)), error) ||
      !CheckCuda(cudaMalloc(&device_input_, cols * sizeof(TFloat)), error) ||
      !CheckCuda(cudaMalloc(&device_output_, rows * sizeof(TFloat)), error) ||
      !CheckCuda(cudaMemcpy(device_weights_, packed.data(), packed.size() * sizeof(TFloat),
                            cudaMemcpyHostToDevice),
                 error)) {
    Invalidate();
    return false;
  }
  return true;
#else
  if (error != nullptr) {
    *error = "CUDA inference is not enabled in this build";
  }
  (void)weights;
  return false;
#endif
}

bool CudaMatrix::MatrixDotVector(const GENERIC_2D_ARRAY<TFloat> &weights, const TFloat *input,
                                 TFloat *output, std::string *error) {
#if defined(HAVE_CUDA) && defined(FAST_FLOAT)
  if (!EnsureWeights(weights, error)) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(device_input_, input, cols_ * sizeof(TFloat), cudaMemcpyHostToDevice), error)) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto status =
      cublasSgemv(CudaEnvironment::Instance().handle(), CUBLAS_OP_T, cols_, rows_,
                  &alpha, static_cast<const float *>(device_weights_), cols_,
                  static_cast<const float *>(device_input_), 1, &beta,
                  static_cast<float *>(device_output_), 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (error != nullptr) {
      *error = "cuBLAS SGEMV failed";
    }
    return false;
  }
  if (!CheckCuda(cudaMemcpy(output, device_output_, rows_ * sizeof(TFloat), cudaMemcpyDeviceToHost), error)) {
    return false;
  }
  for (int r = 0; r < rows_; ++r) {
    output[r] += weights[r][cols_];
  }
  return true;
#else
  if (error != nullptr) {
    *error = "CUDA inference is not enabled in this build";
  }
  (void)weights;
  (void)input;
  (void)output;
  return false;
#endif
}

bool CudaInferenceAvailable(std::string *error) {
#if defined(HAVE_CUDA) && defined(FAST_FLOAT)
  return CudaEnvironment::Instance().Available(error);
#else
  if (error != nullptr) {
    *error = "CUDA inference is not enabled in this build";
  }
  return false;
#endif
}

} // namespace tesseract
