#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include "layer.h"

template<typename Real> class SoftmaxOutputLayer : public Layer<Real> {
 public:
  SoftmaxOutputLayer() {}
  SoftmaxOutputLayer(int input_size,
                     int output_size) {
    this->name_ = "SoftmaxOutput";
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~SoftmaxOutputLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  virtual void CreateParameters(Parameters<Real> *parameters) {
    Matrix<Real> *Why, *dWhy;
    ParameterGlorotInitializer<Real> initializer(ActivationFunctions::LOGISTIC);
    parameters->CreateMatrixParameter("Why", output_size_, input_size_,
                                      &initializer,
                                      &Why, &dWhy);
    Vector<Real> *by, *dby;
    parameters->CreateVectorParameter("by", output_size_, &by, &dby);

    SetParameters(Why, by, dWhy, dby);
  }

  void SetParameters(Matrix<Real> *Why,
                     Vector<Real> *by,
                     Matrix<Real> *dWhy,
                     Vector<Real> *dby) {
    Why_ = Why;
    by_ = by;
    dWhy_ = dWhy;
    dby_ = dby;
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(Why_);
    biases->push_back(by_);

    weight_names->push_back("Why");
    bias_names->push_back("by");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(dWhy_);
    bias_derivatives->push_back(dby_);
  }

  void RunForward() {
    const Matrix<Real> &h = this->GetInput();
    assert(h.cols() == 1);
    Vector<Real> y = *Why_ * h + *by_;
    Real logsum = LogSumExp(y);
    // This is the probability vector.
    this->SetOutput((y.array() - logsum).exp());
  }

  void RunBackward() {
    const Matrix<Real> &h = this->GetInput();
    assert(h.cols() == 1);
    Matrix<Real> *dh = this->GetInputDerivative();
    assert(dh->cols() == 1);

    Vector<Real> dy = *(this->GetMutableOutput());
    dy[output_label_] -= 1.0; // Backprop into y (softmax grad).
    dWhy_->noalias() += dy * h.transpose();
    dby_->noalias() += dy;
    (*dh).noalias() += Why_->transpose() * dy; // Backprop into h.
  }

  int output_label() { return output_label_; }
  void set_output_label(int output_label) {
    output_label_ = output_label;
  }

 protected:
  int input_size_;
  int output_size_;

  Matrix<Real> *Why_;
  Vector<Real> *by_;

  Matrix<Real> *dWhy_;
  Vector<Real> *dby_;

  int output_label_; // Output.
};

template<typename Real> class SoftmaxLayer : public Layer<Real> {
 public:
  SoftmaxLayer() {}
  SoftmaxLayer(int input_size,
               int output_size) {
    this->name_ = "Softmax";
    input_size_ = input_size;
    output_size_ = output_size;
    cost_augmented_ = false;
  }
  virtual ~SoftmaxLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  virtual void CreateParameters(Parameters<Real> *parameters) {
    Matrix<Real> *Why, *dWhy;
    ParameterGlorotInitializer<Real> initializer(ActivationFunctions::LOGISTIC);
    parameters->CreateMatrixParameter("Why", output_size_, input_size_,
                                      &initializer,
                                      &Why, &dWhy);
    Vector<Real> *by, *dby;
    parameters->CreateVectorParameter("by", output_size_, &by, &dby);

    SetParameters(Why, by, dWhy, dby);
  }

  void SetParameters(Matrix<Real> *Why,
                     Vector<Real> *by,
                     Matrix<Real> *dWhy,
                     Vector<Real> *dby) {
    Why_ = Why;
    by_ = by;
    dWhy_ = dWhy;
    dby_ = dby;
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(Why_);
    biases->push_back(by_);

    weight_names->push_back("Why");
    bias_names->push_back("by");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(dWhy_);
    bias_derivatives->push_back(dby_);
  }

  void RunForward() {
    const Matrix<Real> &H = this->GetInput();
    int batch_size = H.cols();
    Matrix<Real> Y = (*Why_ * H).colwise() + *by_;
    // If doing cost-augmented decoding (training time),
    // sum the costs to the log-potentials.
    Vector<Real> logsum;
    if (cost_augmented_) {
      // Compute also the probabilities without the costs, which is useful
      // for computing the training accuracy and F1.
      LogSumExp(Y, &logsum);
      // This is the probability vector.
      P_ = (Y.rowwise() - logsum.transpose()).array().exp();
      Y += costs_;
    }
    LogSumExp(Y, &logsum);
    // This is the probability vector.
    this->SetOutput((Y.rowwise() - logsum.transpose()).array().exp());
  }

  void RunBackward() {
    const Matrix<Real> &H = this->GetInput();
    int batch_size = H.cols();
    Matrix<Real> *dH = this->GetInputDerivative();
    assert(dH->cols() == batch_size);

    // This are the label probabilities.
    Matrix<Real> dY = *(this->GetMutableOutput());
    for (int m = 0; m < batch_size; ++m) {
      dY(output_labels_[m], m) -= 1.0; // Backprop into y (softmax grad).
    }
    Vector<Real> dy_sum = dY.rowwise().sum();
    dWhy_->noalias() += dY * H.transpose();
    dby_->noalias() += dy_sum;
    (*dH).noalias() += Why_->transpose() * dY; // Backprop into h.
  }

  const std::vector<int> &output_labels() { return output_labels_; }
  void set_output_labels(const std::vector<int> &output_labels) {
    output_labels_ = output_labels;
  }

  void set_cost_augmented(bool cost_augmented) {
    cost_augmented_ = cost_augmented;
  }
  void set_costs(const Matrix<Real> &costs) {
    costs_ = costs;
  }

  const Matrix<Real> &GetProbabilities() { return P_; }

 protected:
  int input_size_;
  int output_size_;

  Matrix<Real> *Why_;
  Vector<Real> *by_;

  Matrix<Real> *dWhy_;
  Vector<Real> *dby_;

  std::vector<int> output_labels_; // Output.
  bool cost_augmented_; // True for cost-augmented decoding.
  Matrix<Real> costs_; // If cost_augmented_ = true, this contains the costs.
  // If cost_augmented_ = true, P_ contains the unaugmented probabilities.
  Matrix<Real> P_;
};

#endif /* SOFTMAX_LAYER_H_ */
