#ifndef ALIGNMENT_LAYER_H_
#define ALIGNMENT_LAYER_H_

#include "layer.h"

// This layer assumes:
// - An input X (input_size-by-N; the source) whose columns are to be aligned.
// - M target positions.
// - A matrix A (N-by-M) specifying the alignments.
// - The output will be a matrix Y (input_size-by-M) whose columns are
// a transformation (e.g. a weighted average) of the rows in X determined by
// the matrix A: Y = X*A.
template<typename Real> class AlignmentLayer : public Layer<Real> {
 public:
  AlignmentLayer() { this->name_ = "Alignment"; }
  virtual ~AlignmentLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix<Real> *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    const Matrix<Real> &X = this->GetInput();
    this->SetOutput(X * A_);
  }

  void RunBackward() {
    Matrix<Real> *dX = this->GetInputDerivative();
    *dX += this->GetOutputDerivative() * A_.transpose();
  }

  void SetAlignmentMatrix(const Matrix<Real> &A) {
    A_ = A;
  }

 protected:
  Matrix<Real> A_;
};

#endif /* ALIGNMENT_LAYER_H_ */
