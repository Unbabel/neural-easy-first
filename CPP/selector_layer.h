#ifndef SELECTOR_LAYER_H_
#define SELECTOR_LAYER_H_

#include "layer.h"

template<typename Real> class SelectorLayer : public Layer<Real> {
 public:
  SelectorLayer() { this->name_ = "Selector"; }
  virtual ~SelectorLayer() {}

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {}

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    this->SetOutput(x.block(first_row_, first_column_, num_rows_, num_columns_));
  }

  void RunBackward() {
    Matrix<Real> *dx = this->GetInputDerivative();
    (*dx).block(first_row_, first_column_, num_rows_, num_columns_) +=
      this->GetOutputDerivative();
  }

  void DefineBlock(int first_row, int first_column,
                   int num_rows, int num_columns) {
    first_row_ = first_row;
    first_column_ = first_column;
    num_rows_ = num_rows;
    num_columns_ = num_columns;
  }

 protected:
  int first_row_;
  int first_column_;
  int num_rows_;
  int num_columns_;
};

#endif /* SELECTOR_LAYER_H_ */
