#ifndef CONCATENATOR_LAYER_H_
#define CONCATENATOR_LAYER_H_

#include "layer.h"

template<typename Real> class ConcatenatorLayer : public Layer<Real> {
 public:
  ConcatenatorLayer() { this->name_ = "Concatenator"; }
  virtual ~ConcatenatorLayer() {}

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
    int num_rows = 0;
    int num_columns = this->inputs_[0]->cols();
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      assert(this->inputs_[i]->cols() == num_columns);
      num_rows += this->inputs_[i]->rows();
    }
    this->GetMutableOutput()->setZero(num_rows, num_columns);
    int start = 0;
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      this->GetMutableOutput()->block(start, 0, this->inputs_[i]->rows(), num_columns) =
        *(this->inputs_[i]);
      start += this->inputs_[i]->rows();
    }
  }

  void RunBackward() {
    int num_columns = this->GetOutputDerivative().cols();
    int start = 0;
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      *(this->input_derivatives_[i]) +=
        this->GetOutputDerivative().block(start, 0,
                                          this->input_derivatives_[i]->rows(),
                                          num_columns);
      start += this->inputs_[i]->rows();
    }
  }
};

#endif /* CONCATENATOR_LAYER_H_ */
