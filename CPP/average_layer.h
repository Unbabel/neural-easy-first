#ifndef AVERAGE_LAYER_H_
#define AVERAGE_LAYER_H_

#include "layer.h"

template<typename Real> class AverageLayer : public Layer<Real> {
 public:
  AverageLayer() { this->name_ = "Average"; }
  virtual ~AverageLayer() {}

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
    const Matrix<Real> &x = this->GetInput();
    this->SetOutput(x.rowwise().sum() / static_cast<double>(x.cols()));
  }

  void RunBackward() {
    //std::cout << "tmp1" << std::endl;
    Matrix<Real> *dx = this->GetInputDerivative();
    assert(this->GetOutputDerivative().cols() == 1);
    dx->colwise() += this->GetOutputDerivative().col(0) /
      static_cast<double>(dx->cols());
    //std::cout << "tmp2" << std::endl;
  }
};

#endif /* AVERAGE_LAYER_H_ */
