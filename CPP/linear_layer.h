#ifndef LINEAR_LAYER_H_
#define LINEAR_LAYER_H_

#include "layer.h"

template<typename Real> class LinearLayer : public Layer<Real> {
 public:
  LinearLayer(int input_size, int output_size) {
    this->name_ = "Linear";
    input_size_ = input_size;
    output_size_ = output_size;
  }

  virtual ~LinearLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void CreateParameters(Parameters<Real> *parameters) {
    Matrix<Real> *Wxy, *dWxy;
    ParameterGlorotInitializer<Real> initializer(ActivationFunctions::TANH);
    parameters->CreateMatrixParameter("Wxy", output_size_, input_size_,
                                      &initializer,
                                      &Wxy, &dWxy);
    Vector<Real> *by, *dby;
    parameters->CreateVectorParameter("by", output_size_, &by, &dby);
    SetParameters(Wxy, by, dWxy, dby);
  }

  void SetParameters(Matrix<Real> *Wxy, Vector<Real> *by,
                     Matrix<Real> *dWxy, Vector<Real> *dby) {
    Wxy_ = Wxy;
    by_ = by;
    dWxy_ = dWxy;
    dby_ = dby;
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    // Remove this function?
    weights->push_back(Wxy_);
    weight_names->push_back("linear_weights");

    biases->push_back(by_);
    bias_names->push_back("linear_bias");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    // Remove this function?
    weight_derivatives->push_back(dWxy_);
    bias_derivatives->push_back(dby_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff = 1.0; // Like in TANH.
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    this->SetOutput((*Wxy_ * x).colwise() + *by_);
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();
    (*dx).noalias() += Wxy_->transpose() * this->GetOutputDerivative();
    dWxy_->noalias() += this->GetOutputDerivative() * x.transpose();
    dby_->noalias() += this->GetOutputDerivative().rowwise().sum();
  }

 protected:
  int input_size_;
  int output_size_;
  Matrix<Real> *Wxy_;
  Vector<Real> *by_;

  Matrix<Real> *dWxy_;
  Vector<Real> *dby_;
};

#endif /* LINEAR_LAYER_H_ */
