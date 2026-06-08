///////////////////////////////////////////////////////////////////////
// File:        cudabackend.h
// Description: Optional CUDA helper for LSTM inference.
// Author:      GitHub Copilot
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
///////////////////////////////////////////////////////////////////////

#ifndef TESSERACT_LSTM_CUDABACKEND_H_
#define TESSERACT_LSTM_CUDABACKEND_H_

#include "matrix.h"
#include "tesstypes.h"

#include <string>

namespace tesseract {

class CudaMatrix {
public:
  CudaMatrix();
  ~CudaMatrix();

  bool Prepare(const GENERIC_2D_ARRAY<TFloat> &weights, std::string *error);
  bool MatrixDotVector(const GENERIC_2D_ARRAY<TFloat> &weights, const TFloat *input, TFloat *output,
                       std::string *error);
  void Invalidate();

private:
  bool EnsureWeights(const GENERIC_2D_ARRAY<TFloat> &weights, std::string *error);

  int rows_;
  int cols_;
  void *device_weights_;
  void *device_input_;
  void *device_output_;
};

bool CudaInferenceAvailable(std::string *error);

} // namespace tesseract

#endif // TESSERACT_LSTM_CUDABACKEND_H_
