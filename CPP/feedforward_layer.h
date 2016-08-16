#ifndef FEEDFORWARD_LAYER_H_
#define FEEDFORWARD_LAYER_H_

#include "layer.h"

template<typename Real> class FeedforwardLayer : public Layer<Real> {
 public:
  FeedforwardLayer() {}
  FeedforwardLayer(int input_size,
                   int output_size) {
    this->name_ = "Feedforward";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~FeedforwardLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  virtual void CreateParameters(Parameters<Real> *parameters) {
    Matrix<Real> *Wxh, *dWxh;
    ParameterGlorotInitializer<Real> initializer(activation_function_);
    parameters->CreateMatrixParameter("Wxh", output_size_, input_size_,
                                      &initializer,
                                      &Wxh, &dWxh);
    Vector<Real> *bh, *dbh;
    parameters->CreateVectorParameter("bh", output_size_, &bh, &dbh);

    SetParameters(Wxh, bh, dWxh, dbh);
  }

  void SetParameters(Matrix<Real> *Wxh,
                     Vector<Real> *bh,
                     Matrix<Real> *dWxh,
                     Vector<Real> *dbh) {
    Wxh_ = Wxh;
    bh_ = bh;
    dWxh_ = dWxh;
    dbh_ = dbh;
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(Wxh_);
    biases->push_back(bh_);

    weight_names->push_back("Wxh");
    bias_names->push_back("bh");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(dWxh_);
    bias_derivatives->push_back(dbh_);
  }

  void RunForward() {
    // In the normal case, x is a column vector.
    // If x has several columns, one assumes a feedforward net
    // for every column with shared parameters.
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> tmp = (*Wxh_ * x).colwise() + *bh_;
    EvaluateActivation(activation_function_,
                       tmp,
                       this->GetMutableOutput());
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();
    Matrix<Real> dhraw;
    DerivateActivation(activation_function_, *(this->GetMutableOutput()),
                       &dhraw);
    dhraw = dhraw.array() * this->GetOutputDerivative().array();
    dWxh_->noalias() += dhraw * x.transpose();
    dbh_->noalias() += dhraw.rowwise().sum();
    *dx += Wxh_->transpose() * dhraw; // Backprop into x.
  }

 protected:
  int activation_function_;
  int output_size_;
  int input_size_;

  Matrix<Real> *Wxh_;
  Vector<Real> *bh_;

  Matrix<Real> *dWxh_;
  Vector<Real> *dbh_;
};

#endif /* FEEDFORWARD_LAYER_H_ */
