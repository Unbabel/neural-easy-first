#ifndef RNN_LAYER_H_
#define RNN_LAYER_H_

#include "layer.h"

template<typename Real> class RNNLayer : public Layer<Real> {
 public:
  RNNLayer() {}
  RNNLayer(int input_size,
           int hidden_size) {
    this->name_ = "RNN";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
    use_control_ = false;
  }
  virtual ~RNNLayer() {}

  int input_size() const { return input_size_; }
  int hidden_size() const { return hidden_size_; }

  void set_use_control(bool use_control) { use_control_ = use_control; }

  virtual void ResetParameters() {
    Wxh_ = Matrix<Real>::Zero(hidden_size_, input_size_);
    Whh_ = Matrix<Real>::Zero(hidden_size_, hidden_size_);
    bh_ = Vector<Real>::Zero(hidden_size_);
    if (use_hidden_start_ && !use_control_) {
      h0_ = Vector<Real>::Zero(hidden_size_);
    }
  }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    weights->push_back(&Wxh_);
    weights->push_back(&Whh_);

    biases->push_back(&bh_);
    if (use_hidden_start_ && !use_control_) {
      biases->push_back(&h0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxh");
    weight_names->push_back("Whh");

    bias_names->push_back("bh");
    if (use_hidden_start_ && !use_control_) {
      bias_names->push_back("h0");
    }
  }

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxh_);
    weight_derivatives->push_back(&dWhh_);
    bias_derivatives->push_back(&dbh_);
    if (use_hidden_start_ && !use_control_) {
      bias_derivatives->push_back(&dh0_); // Not really a bias, but goes here.
    }
  }

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    if (activation_function_ == ActivationFunctions::LOGISTIC) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  virtual void ResetGradients() {
    dWxh_.setZero(hidden_size_, input_size_);
    dbh_.setZero(hidden_size_);
    dWhh_.setZero(hidden_size_, hidden_size_);
    if (use_hidden_start_ && !use_control_) {
      dh0_.setZero(hidden_size_);
    }
  }

  virtual void RunForward() {
    const Matrix<Real> &x = *(this->inputs_[0]);
    int length = x.cols();
    this->GetMutableOutput()->setZero(hidden_size_, length);
    Matrix<Real> hraw = (Wxh_ * x).colwise() + bh_;
    Vector<Real> hprev = Vector<Real>::Zero(this->GetMutableOutput()->rows());
    if (use_hidden_start_) {
      if (!use_control_) {
        hprev = h0_;
      } else {
        hprev = *(this->inputs_[1]);
      }
    }
    Vector<Real> result;
    for (int t = 0; t < length; ++t) {
      Vector<Real> tmp = hraw.col(t) + Whh_ * hprev;
      EvaluateActivation(activation_function_,
                         tmp,
                         &result);
      this->GetMutableOutput()->col(t) = result;
      hprev = this->GetMutableOutput()->col(t);
    }
  }

  virtual void RunBackward() {
    const Matrix<Real> &x = *(this->inputs_[0]);
    Matrix<Real> *dx = this->input_derivatives_[0];

    Vector<Real> dhnext = Vector<Real>::Zero(Whh_.rows());
    const Matrix<Real> &dy = this->GetOutputDerivative();

    Matrix<Real> dhraw;
    DerivateActivation(activation_function_, *(this->GetMutableOutput()), &dhraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = dy.col(t) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = Whh_.transpose() * dhraw.col(t);
    }

    *dx += Wxh_.transpose() * dhraw; // Backprop into x.

    dWxh_.noalias() += dhraw * x.transpose();
    dbh_.noalias() += dhraw.rowwise().sum();
    dWhh_.noalias() += dhraw.rightCols(length-1) *
      this->GetMutableOutput()->leftCols(length-1).transpose();
    if (use_hidden_start_) {
      if (!use_control_) {
        dh0_.noalias() += dhnext;
      } else {
        *(this->input_derivatives_[1]) += dhnext;
      }
    }
  }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  bool use_hidden_start_;
  bool use_control_;

  Matrix<Real> Wxh_;
  Matrix<Real> Whh_;
  Vector<Real> bh_;
  Vector<Real> h0_;

  Matrix<Real> dWxh_;
  Matrix<Real> dWhh_;
  Vector<Real> dbh_;
  Vector<Real> dh0_;
};

template<typename Real> class BiRNNLayer : public RNNLayer<Real> {
 public:
  BiRNNLayer() {}
  BiRNNLayer(int input_size,
           int hidden_size) {
    this->name_ = "BiRNN";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
    this->use_control_ = false; // TODO: handle control state.
  }
  virtual ~BiRNNLayer() {}

  void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxl_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wll_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bl_ = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      l0_ = Vector<Real>::Zero(this->hidden_size_);
    }
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    RNNLayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);

    biases->push_back(&bl_);
    if (this->use_hidden_start_) {
      biases->push_back(&l0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");

    bias_names->push_back("bl");
    if (this->use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    RNNLayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxl_);
    weight_derivatives->push_back(&dWll_);
    bias_derivatives->push_back(&dbl_);
    if (this->use_hidden_start_) {
      bias_derivatives->push_back(&dl0_); // Not really a bias, but goes here.
    }
  }

  void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxl_.setZero(this->hidden_size_, this->input_size_);
    dbl_.setZero(this->hidden_size_);
    dWll_.setZero(this->hidden_size_, this->hidden_size_);
    if (this->use_hidden_start_) {
      dl0_.setZero(this->hidden_size_);
    }
  }

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    int length = x.cols();
    this->GetMutableOutput()->setZero(2*this->hidden_size_, length);
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Matrix<Real> lraw = (Wxl_ * x).colwise() + bl_;
    Vector<Real> hprev = Vector<Real>::Zero(this->hidden_size_);
    Vector<Real> lnext = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      hprev = this->h0_;
      lnext = l0_;
    }
    Vector<Real> result;
    for (int t = 0; t < length; ++t) {
      Vector<Real> tmp = hraw.col(t) + this->Whh_ * hprev;
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->GetMutableOutput()->block(0, t, this->hidden_size_, 1) = result;
      hprev = result;
    }
    for (int t = length-1; t >= 0; --t) {
      Vector<Real> tmp = lraw.col(t) + Wll_ * lnext;
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->GetMutableOutput()->block(this->hidden_size_, t, this->hidden_size_, 1) = result;
      lnext = result;
    }
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    Vector<Real> dlprev = Vector<Real>::Zero(Wll_.rows());

    int length = this->GetMutableOutput()->cols();
    Matrix<Real> result;
    DerivateActivation(this->activation_function_, this->output_, &result);
    Matrix<Real> dhraw = result.block(0, 0, this->hidden_size_, length);
    Matrix<Real> dlraw = result.block(this->hidden_size_, 0, this->hidden_size_, length);

    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = this->GetOutputDerivative().block(0, t, this->hidden_size_, 1) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = this->Whh_.transpose() * dhraw.col(t);
    }

    for (int t = 0; t < length; ++t) {
      Vector<Real> dl = this->GetOutputDerivative().block(this->hidden_size_, t, this->hidden_size_, 1) + dlprev; // Backprop into h.
      dlraw.col(t) = dlraw.col(t).array() * dl.array();
      dlprev.noalias() = Wll_.transpose() * dlraw.col(t);
    }

    *dx += this->Wxh_.transpose() * dhraw; // Backprop into x.
    *dx += Wxl_.transpose() * dlraw; // Backprop into x.

    this->dWxh_.noalias() += dhraw * x.transpose();
    this->dbh_.noalias() += dhraw.rowwise().sum();
    this->dWhh_.noalias() += dhraw.rightCols(length-1) *
      this->GetMutableOutput()->block(0, 0, this->hidden_size_, length-1).transpose();
    this->dh0_.noalias() += dhnext;

    dWxl_.noalias() += dlraw * x.transpose();
    dbl_.noalias() += dlraw.rowwise().sum();
    dWll_.noalias() += dlraw.leftCols(length-1) *
      this->GetMutableOutput()->block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dl0_.noalias() += dlprev;
  }

 protected:
  Matrix<Real> Wxl_;
  Matrix<Real> Wll_;
  Vector<Real> bl_;
  Vector<Real> l0_;

  Matrix<Real> dWxl_;
  Matrix<Real> dWll_;
  Vector<Real> dbl_;
  Vector<Real> dl0_;
};

template<typename Real> class GRULayer : public RNNLayer<Real> {
 public:
  GRULayer() {}
  GRULayer(int input_size,
           int hidden_size) {
    this->name_ = "GRU";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
    this->use_control_ = false;
  }
  virtual ~GRULayer() {}

  virtual void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxz_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whz_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxr_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whr_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bz_ = Vector<Real>::Zero(this->hidden_size_);
    br_ = Vector<Real>::Zero(this->hidden_size_);
  }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    RNNLayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);

    weights->push_back(&Wxz_);
    weights->push_back(&Whz_);
    weights->push_back(&Wxr_);
    weights->push_back(&Whr_);

    biases->push_back(&bz_);
    biases->push_back(&br_);

    weight_names->push_back("Wxz");
    weight_names->push_back("Whz");
    weight_names->push_back("Wxr");
    weight_names->push_back("Whr");

    bias_names->push_back("bz");
    bias_names->push_back("br");
  }

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    RNNLayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWhz_);
    weight_derivatives->push_back(&dWxr_);
    weight_derivatives->push_back(&dWhr_);
    bias_derivatives->push_back(&dbz_);
    bias_derivatives->push_back(&dbr_);
  }

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (this->activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &Wxz_ || W == &Whz_ || W == &Wxr_ || W == &Whr_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  virtual void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxz_.setZero(this->hidden_size_, this->input_size_);
    dWhz_.setZero(this->hidden_size_, this->hidden_size_);
    dWxr_.setZero(this->hidden_size_, this->input_size_);
    dWhr_.setZero(this->hidden_size_, this->hidden_size_);
    dbz_.setZero(this->hidden_size_);
    dbr_.setZero(this->hidden_size_);
  }

  virtual void RunForward() {
    assert(this->GetNumInputs() == 1 || this->use_control_);
    const Matrix<Real> &x = *(this->inputs_[0]);

    int length = x.cols();
    z_.setZero(this->hidden_size_, length);
    r_.setZero(this->hidden_size_, length);
    hu_.setZero(this->hidden_size_, length);
    this->GetMutableOutput()->setZero(this->hidden_size_, length);
    Matrix<Real> zraw = (Wxz_ * x).colwise() + bz_;
    Matrix<Real> rraw = (Wxr_ * x).colwise() + br_;
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Vector<Real> hprev = Vector<Real>::Zero(this->GetMutableOutput()->rows());
    if (this->use_hidden_start_) {
      if (!this->use_control_) {
        hprev = this->h0_;
      } else {
        hprev = *(this->inputs_[1]);
      }
    }
    Vector<Real> result;
    Vector<Real> tmp;
    for (int t = 0; t < length; ++t) {
      tmp = zraw.col(t) + Whz_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      z_.col(t) = result;

      tmp = rraw.col(t) + Whr_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      r_.col(t) = result;

      tmp = hraw.col(t) + this->Whh_ * r_.col(t).cwiseProduct(hprev);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      hu_.col(t) = result;
      this->GetMutableOutput()->col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;
      hprev = this->GetMutableOutput()->col(t);
    }
  }

  virtual void RunBackward() {
    const Matrix<Real> &x = *(this->inputs_[0]);
    Matrix<Real> *dx = this->input_derivatives_[0];

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    const Matrix<Real> &dy = this->GetOutputDerivative();

    Matrix<Real> dhuraw;
    DerivateActivation(this->activation_function_, hu_, &dhuraw);
    Matrix<Real> dzraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, z_, &dzraw);
    Matrix<Real> drraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, r_, &drraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = dy.col(t) + dhnext; // Backprop into h.
      Vector<Real> dhu = z_.col(t).cwiseProduct(dh);

      dhuraw.col(t) = dhuraw.col(t).cwiseProduct(dhu);
      Vector<Real> hprev;
      if (t == 0) {
        // hprev = Vector<Real>::Zero(this->GetMutableOutput()->rows()); <- LOOKS LIKE A BUG.
        hprev = Vector<Real>::Zero(this->GetMutableOutput()->rows());
        if (this->use_hidden_start_) {
          if (!this->use_control_) {
            hprev = this->h0_;
          } else {
            hprev = *(this->inputs_[1]);
          }
        }
      } else {
        hprev = this->GetMutableOutput()->col(t-1);
      }

      Vector<Real> dq = this->Whh_.transpose() * dhuraw.col(t);
      Vector<Real> dz = (hu_.col(t) - hprev).cwiseProduct(dh);
      Vector<Real> dr = hprev.cwiseProduct(dq);

      dzraw.col(t) = dzraw.col(t).cwiseProduct(dz);
      drraw.col(t) = drraw.col(t).cwiseProduct(dr);

      dhnext.noalias() =
        Whz_.transpose() * dzraw.col(t) +
        Whr_.transpose() * drraw.col(t) +
        r_.col(t).cwiseProduct(dq) +
        (1.0 - z_.col(t).array()).matrix().cwiseProduct(dh);
    }

    *dx += Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
      this->Wxh_.transpose() * dhuraw; // Backprop into x.

    dWxz_.noalias() += dzraw * x.transpose();
    dbz_.noalias() += dzraw.rowwise().sum();
    dWxr_.noalias() += drraw * x.transpose();
    dbr_.noalias() += drraw.rowwise().sum();
    this->dWxh_.noalias() += dhuraw * x.transpose();
    this->dbh_.noalias() += dhuraw.rowwise().sum();

    dWhz_.noalias() += dzraw.rightCols(length-1) *
      this->GetMutableOutput()->leftCols(length-1).transpose();
    dWhr_.noalias() += drraw.rightCols(length-1) *
      this->GetMutableOutput()->leftCols(length-1).transpose();
    this->dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((r_.rightCols(length-1)).cwiseProduct(this->GetMutableOutput()->leftCols(length-1))).
      transpose();

    if (this->use_hidden_start_) {
      if (!this->use_control_) {
        this->dh0_.noalias() += dhnext;
      } else {
        *(this->input_derivatives_[1]) += dhnext;
      }
    }
  }

 protected:
  Matrix<Real> Wxz_;
  Matrix<Real> Whz_;
  Matrix<Real> Wxr_;
  Matrix<Real> Whr_;
  Vector<Real> bz_;
  Vector<Real> br_;

  Matrix<Real> dWxz_;
  Matrix<Real> dWhz_;
  Matrix<Real> dWxr_;
  Matrix<Real> dWhr_;
  Vector<Real> dbz_;
  Vector<Real> dbr_;

  Matrix<Real> z_;
  Matrix<Real> r_;
  Matrix<Real> hu_;
};

template<typename Real> class LSTMLayer : public RNNLayer<Real> {
 public:
  LSTMLayer() {}
  LSTMLayer(int input_size,
            int hidden_size) {
    this->name_ = "LSTM";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
    this->use_control_ = false;
  }
  virtual ~LSTMLayer() {}

  virtual void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxi_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whi_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxf_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whf_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxo_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Who_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bi_ = Vector<Real>::Zero(this->hidden_size_);
    bf_ = Vector<Real>::Zero(this->hidden_size_);
    bo_ = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_ && !this->use_control_) {
      c0_ = Vector<Real>::Zero(this->hidden_size_);
    }
  }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    RNNLayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);
    weights->push_back(&Wxi_);
    weights->push_back(&Whi_);
    weights->push_back(&Wxf_);
    weights->push_back(&Whf_);
    weights->push_back(&Wxo_);
    weights->push_back(&Who_);

    biases->push_back(&bi_);
    biases->push_back(&bf_);
    biases->push_back(&bo_);
    if (this->use_hidden_start_ && !this->use_control_) {
      biases->push_back(&c0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxi");
    weight_names->push_back("Whi");
    weight_names->push_back("Wxf");
    weight_names->push_back("Whf");
    weight_names->push_back("Wxo");
    weight_names->push_back("Who");

    bias_names->push_back("bi");
    bias_names->push_back("bf");
    bias_names->push_back("bo");
    if (this->use_hidden_start_ && !this->use_control_) {
      bias_names->push_back("c0");
    }
  }

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    RNNLayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxi_);
    weight_derivatives->push_back(&dWhi_);
    weight_derivatives->push_back(&dWxf_);
    weight_derivatives->push_back(&dWhf_);
    weight_derivatives->push_back(&dWxo_);
    weight_derivatives->push_back(&dWho_);
    bias_derivatives->push_back(&dbi_);
    bias_derivatives->push_back(&dbf_);
    bias_derivatives->push_back(&dbo_);
    if (this->use_hidden_start_ && !this->use_control_) {
      bias_derivatives->push_back(&dc0_); // Not really a bias, but it goes here.
    }
  }

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (this->activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &Wxi_ || W == &Whi_ || W == &Wxf_ || W == &Whf_ ||
        W == &Wxo_ || W == &Who_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  virtual void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxi_.setZero(this->hidden_size_, this->input_size_);
    dWhi_.setZero(this->hidden_size_, this->hidden_size_);
    dWxf_.setZero(this->hidden_size_, this->input_size_);
    dWhf_.setZero(this->hidden_size_, this->hidden_size_);
    dWxo_.setZero(this->hidden_size_, this->input_size_);
    dWho_.setZero(this->hidden_size_, this->hidden_size_);
    dbi_.setZero(this->hidden_size_);
    dbf_.setZero(this->hidden_size_);
    dbo_.setZero(this->hidden_size_);
    if (this->use_hidden_start_ && !this->use_control_) {
      dc0_.setZero(this->hidden_size_);
    }
  }

  virtual void RunForward() {
    assert(this->GetNumInputs() == 1 || this->use_control_);
    const Matrix<Real> &x = *(this->inputs_[0]);

    int length = x.cols();
    i_.setZero(this->hidden_size_, length);
    f_.setZero(this->hidden_size_, length);
    o_.setZero(this->hidden_size_, length);
    cu_.setZero(this->hidden_size_, length);
    c_.setZero(this->hidden_size_, length);
    hu_.setZero(this->hidden_size_, length);
    this->outputs_[0].setZero(this->hidden_size_, length);
    Matrix<Real> iraw = (Wxi_ * x).colwise() + bi_;
    Matrix<Real> fraw = (Wxf_ * x).colwise() + bf_;
    Matrix<Real> oraw = (Wxo_ * x).colwise() + bo_;
    Matrix<Real> curaw = (this->Wxh_ * x).colwise() + this->bh_;
    Vector<Real> hprev = Vector<Real>::Zero(this->hidden_size_);
    Vector<Real> cprev = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      if (!this->use_control_) {
        hprev = this->h0_;
        cprev = this->c0_;
      } else {
        hprev = Vector<Real>::Zero(this->hidden_size_);
        cprev = *(this->inputs_[1]);
      }
    }
    Vector<Real> result;
    Vector<Real> tmp;
    for (int t = 0; t < length; ++t) {
      tmp = iraw.col(t) + Whi_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      i_.col(t) = result;

      tmp = fraw.col(t) + Whf_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      f_.col(t) = result;

      tmp = oraw.col(t) + Who_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      o_.col(t) = result;

      tmp = curaw.col(t) + this->Whh_ * hprev;
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      cu_.col(t) = result;

      c_.col(t) = f_.col(t).cwiseProduct(cprev) + i_.col(t).cwiseProduct(cu_.col(t));

      tmp = c_.col(t);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      hu_.col(t) = result;

      this->outputs_[0].col(t) = o_.col(t).cwiseProduct(hu_.col(t));

      hprev = this->outputs_[0].col(t);
      cprev = c_.col(t);
    }
    this->outputs_[1] = c_; // TODO: don't duplicate these variables.
  }

  virtual void RunBackward() {
    const Matrix<Real> &x = *(this->inputs_[0]);
    Matrix<Real> *dx = this->input_derivatives_[0];

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    Vector<Real> dcnext = Vector<Real>::Zero(this->hidden_size_);
    //Vector<Real> fnext = Vector<Real>::Zero(this->hidden_size_);
    const Matrix<Real> &dy = this->output_derivatives_[0];

    Matrix<Real> dcuraw;
    DerivateActivation(this->activation_function_, cu_, &dcuraw);
    Matrix<Real> dhuraw;
    DerivateActivation(this->activation_function_, hu_, &dhuraw);
    Matrix<Real> diraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, i_, &diraw);
    Matrix<Real> dfraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, f_, &dfraw);
    Matrix<Real> doraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, o_, &doraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = dy.col(t) + dhnext; // Backprop into h.
      Vector<Real> dhu = o_.col(t).cwiseProduct(dh);
      Vector<Real> dout = hu_.col(t).cwiseProduct(dh);

      Vector<Real> dc = this->output_derivatives_[1].col(t);
      dc += dhuraw.col(t).cwiseProduct(dhu) + dcnext;

      Vector<Real> cprev;
      if (t == 0) {
        if (this->use_hidden_start_) {
          if (!this->use_control_) {
            cprev = c0_;
          } else {
            cprev = *(this->inputs_[1]);
          }
        } else {
          cprev = Vector<Real>::Zero(c_.rows());
        }
      } else {
        cprev = this->c_.col(t-1);
      }

      Vector<Real> di = cu_.col(t).cwiseProduct(dc);
      Vector<Real> df = cprev.cwiseProduct(dc);
      Vector<Real> dcu = i_.col(t).cwiseProduct(dc);

      diraw.col(t) = diraw.col(t).cwiseProduct(di);
      dfraw.col(t) = dfraw.col(t).cwiseProduct(df);
      doraw.col(t) = doraw.col(t).cwiseProduct(dout);
      dcuraw.col(t) = dcuraw.col(t).cwiseProduct(dcu);

      dhnext.noalias() =
        Who_.transpose() * doraw.col(t) +
        Whf_.transpose() * dfraw.col(t) +
        Whi_.transpose() * diraw.col(t) +
        this->Whh_.transpose() * dcuraw.col(t);

      dcnext.noalias() = dc.cwiseProduct(f_.col(t));
      //fnext.noalias() = f_.col(t);
    }

    *dx += Wxo_.transpose() * doraw + Wxf_.transpose() * dfraw +
      Wxi_.transpose() * diraw + this->Wxh_.transpose() * dcuraw; // Backprop into x.

    dWxi_.noalias() += diraw * x.transpose();
    dbi_.noalias() += diraw.rowwise().sum();
    dWxf_.noalias() += dfraw * x.transpose();
    dbf_.noalias() += dfraw.rowwise().sum();
    dWxo_.noalias() += doraw * x.transpose();
    dbo_.noalias() += doraw.rowwise().sum();
    this->dWxh_.noalias() += dcuraw * x.transpose();
    this->dbh_.noalias() += dcuraw.rowwise().sum();

    dWhi_.noalias() += diraw.rightCols(length-1) *
      this->outputs_[0].leftCols(length-1).transpose();
    dWhf_.noalias() += dfraw.rightCols(length-1) *
      this->outputs_[0].leftCols(length-1).transpose();
    dWho_.noalias() += doraw.rightCols(length-1) *
      this->outputs_[0].leftCols(length-1).transpose();
    this->dWhh_.noalias() += dcuraw.rightCols(length-1) *
      this->outputs_[0].leftCols(length-1).transpose();

    if (this->use_hidden_start_) {
      if (!this->use_control_) {
        this->dh0_.noalias() += dhnext;
        this->dc0_.noalias() += dcnext;
      } else {
        *(this->input_derivatives_[1]) += dcnext;
      }
    }
  }

 protected:
  Matrix<Real> Wxi_;
  Matrix<Real> Whi_;
  Matrix<Real> Wxf_;
  Matrix<Real> Whf_;
  Matrix<Real> Wxo_;
  Matrix<Real> Who_;
  Vector<Real> bi_;
  Vector<Real> bf_;
  Vector<Real> bo_;
  Vector<Real> c0_;

  Matrix<Real> dWxi_;
  Matrix<Real> dWhi_;
  Matrix<Real> dWxf_;
  Matrix<Real> dWhf_;
  Matrix<Real> dWxo_;
  Matrix<Real> dWho_;
  Vector<Real> dbi_;
  Vector<Real> dbf_;
  Vector<Real> dbo_;
  Vector<Real> dc0_;

  Matrix<Real> i_;
  Matrix<Real> f_;
  Matrix<Real> o_;
  Matrix<Real> c_;
  Matrix<Real> cu_;
  Matrix<Real> hu_;
};

template<typename Real> class BiGRULayer : public GRULayer<Real> {
 public:
  BiGRULayer() {}
  BiGRULayer(int input_size,
             int hidden_size) {
    this->name_ = "BiGRU";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
    this->use_control_ = false;
  }
  virtual ~BiGRULayer() {}

  void ResetParameters() {
    GRULayer<Real>::ResetParameters();

    Wxl_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wll_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxz_r_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wlz_r_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxr_r_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wlr_r_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bl_ = Vector<Real>::Zero(this->hidden_size_);
    bz_r_ = Vector<Real>::Zero(this->hidden_size_);
    br_r_ = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      l0_ = Vector<Real>::Zero(this->hidden_size_);
    }
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    GRULayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);
    weights->push_back(&Wxz_r_);
    weights->push_back(&Wlz_r_);
    weights->push_back(&Wxr_r_);
    weights->push_back(&Wlr_r_);

    biases->push_back(&bl_);
    biases->push_back(&bz_r_);
    biases->push_back(&br_r_);
    if (this->use_hidden_start_) {
      biases->push_back(&l0_);
    }

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");
    weight_names->push_back("Wxz_r");
    weight_names->push_back("Wlz_r");
    weight_names->push_back("Wxr_r");
    weight_names->push_back("Wlr_r");

    bias_names->push_back("bl");
    bias_names->push_back("bz_r");
    bias_names->push_back("br_r");
    if (this->use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    GRULayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxl_);
    weight_derivatives->push_back(&dWll_);
    weight_derivatives->push_back(&dWxz_r_);
    weight_derivatives->push_back(&dWlz_r_);
    weight_derivatives->push_back(&dWxr_r_);
    weight_derivatives->push_back(&dWlr_r_);
    bias_derivatives->push_back(&dbl_);
    bias_derivatives->push_back(&dbz_r_);
    bias_derivatives->push_back(&dbr_r_);
    if (this->use_hidden_start_) {
      bias_derivatives->push_back(&dl0_);
    }
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (this->activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &this->Wxz_ || W == &this->Whz_ || W == &this->Wxr_ || W == &this->Whr_ ||
        W == &this->Wxz_r_ || W == &this->Wlz_r_ || W == &this->Wxr_r_ || W == &this->Wlr_r_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    GRULayer<Real>::ResetGradients();
    dWxl_.setZero(this->hidden_size_, this->input_size_);
    dWll_.setZero(this->hidden_size_, this->hidden_size_);
    dWxz_r_.setZero(this->hidden_size_, this->input_size_);
    dWlz_r_.setZero(this->hidden_size_, this->hidden_size_);
    dWxr_r_.setZero(this->hidden_size_, this->input_size_);
    dWlr_r_.setZero(this->hidden_size_, this->hidden_size_);
    dbl_.setZero(this->hidden_size_);
    dbz_r_.setZero(this->hidden_size_);
    dbr_r_.setZero(this->hidden_size_);
    if (this->use_hidden_start_) {
      dl0_.setZero(this->hidden_size_);
    }
  }

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();

    int length = x.cols();
    this->z_.setZero(this->hidden_size_, length);
    this->r_.setZero(this->hidden_size_, length);
    this->hu_.setZero(this->hidden_size_, length);
    z_r_.setZero(this->hidden_size_, length);
    r_r_.setZero(this->hidden_size_, length);
    lu_.setZero(this->hidden_size_, length);

    this->GetMutableOutput()->setZero(2*this->hidden_size_, length);

    Matrix<Real> zraw = (this->Wxz_ * x).colwise() + this->bz_;
    Matrix<Real> rraw = (this->Wxr_ * x).colwise() + this->br_;
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Matrix<Real> zraw_r = (Wxz_r_ * x).colwise() + bz_r_;
    Matrix<Real> rraw_r = (Wxr_r_ * x).colwise() + br_r_;
    Matrix<Real> lraw = (Wxl_ * x).colwise() + bl_;
    Vector<Real> hprev = Vector<Real>::Zero(this->hidden_size_);
    Vector<Real> lnext = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      hprev = this->h0_;
      lnext = l0_;
    }
    Vector<Real> result;
    Vector<Real> tmp;
    for (int t = 0; t < length; ++t) {
      tmp = zraw.col(t) + this->Whz_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      this->z_.col(t) = result;

      tmp = rraw.col(t) + this->Whr_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      this->r_.col(t) = result;

      tmp = hraw.col(t) + this->Whh_ * this->r_.col(t).cwiseProduct(hprev);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->hu_.col(t) = result;
      this->GetMutableOutput()->block(0, t, this->hidden_size_, 1) =
        this->z_.col(t).cwiseProduct(this->hu_.col(t) - hprev) + hprev;
      hprev = this->GetMutableOutput()->block(0, t, this->hidden_size_, 1);
    }
    for (int t = length-1; t >= 0; --t) {
      tmp = zraw_r.col(t) + Wlz_r_ * lnext;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      z_r_.col(t) = result;

      tmp = rraw_r.col(t) + Wlr_r_ * lnext;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      r_r_.col(t) = result;

      tmp = lraw.col(t) + Wll_ * r_r_.col(t).cwiseProduct(lnext);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      lu_.col(t) = result;
      this->GetMutableOutput()->block(this->hidden_size_, t, this->hidden_size_, 1) =
        z_r_.col(t).cwiseProduct(lu_.col(t) - lnext) + lnext;
      lnext = this->GetMutableOutput()->block(this->hidden_size_, t, this->hidden_size_, 1);
    }
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    Vector<Real> dlprev = Vector<Real>::Zero(Wll_.rows());
    //const Matrix<Real> &dy = this->output_derivative_;

    int length = this->GetMutableOutput()->cols();
    Matrix<Real> dhuraw;
    DerivateActivation(this->activation_function_, this->hu_, &dhuraw);
    Matrix<Real> dzraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, this->z_, &dzraw);
    Matrix<Real> drraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, this->r_, &drraw);
    Matrix<Real> dluraw;
    DerivateActivation(this->activation_function_, lu_, &dluraw);
    Matrix<Real> dzraw_r;
    DerivateActivation(ActivationFunctions::LOGISTIC, z_r_, &dzraw_r);
    Matrix<Real> drraw_r;
    DerivateActivation(ActivationFunctions::LOGISTIC, r_r_, &drraw_r);

    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = this->GetOutputDerivative().block(0, t, this->hidden_size_, 1) + dhnext; // Backprop into h.
      Vector<Real> dhu = this->z_.col(t).cwiseProduct(dh);

      dhuraw.col(t) = dhuraw.col(t).cwiseProduct(dhu);
      Vector<Real> hprev;
      if (t == 0) {
        hprev = Vector<Real>::Zero(this->hidden_size_);
      } else {
        hprev = this->GetMutableOutput()->block(0, t-1, this->hidden_size_, 1);
      }

      Vector<Real> dq = this->Whh_.transpose() * dhuraw.col(t);
      Vector<Real> dz = (this->hu_.col(t) - hprev).cwiseProduct(dh);
      Vector<Real> dr = hprev.cwiseProduct(dq);

      dzraw.col(t) = dzraw.col(t).cwiseProduct(dz);
      drraw.col(t) = drraw.col(t).cwiseProduct(dr);

      dhnext.noalias() =
        this->Whz_.transpose() * dzraw.col(t) +
        this->Whr_.transpose() * drraw.col(t) +
        this->r_.col(t).cwiseProduct(dq) +
        (1.0 - this->z_.col(t).array()).matrix().cwiseProduct(dh);
    }

    for (int t = 0; t < length; ++t) {
      Vector<Real> dl = this->GetOutputDerivative().block(this->hidden_size_, t, this->hidden_size_, 1) + dlprev; // Backprop into h.
      Vector<Real> dlu = z_r_.col(t).cwiseProduct(dl);

      dluraw.col(t) = dluraw.col(t).cwiseProduct(dlu);
      Vector<Real> lnext;
      if (t == length-1) {
        lnext = Vector<Real>::Zero(this->hidden_size_);
      } else {
        lnext = this->GetMutableOutput()->block(this->hidden_size_, t+1, this->hidden_size_, 1);
      }

      Vector<Real> dq_r = this->Wll_.transpose() * dluraw.col(t);
      Vector<Real> dz_r = (lu_.col(t) - lnext).cwiseProduct(dl);
      Vector<Real> dr_r = lnext.cwiseProduct(dq_r);

      dzraw_r.col(t) = dzraw_r.col(t).cwiseProduct(dz_r);
      drraw_r.col(t) = drraw_r.col(t).cwiseProduct(dr_r);

      dlprev.noalias() =
        Wlz_r_.transpose() * dzraw_r.col(t) +
        Wlr_r_.transpose() * drraw_r.col(t) +
        r_r_.col(t).cwiseProduct(dq_r) +
        (1.0 - z_r_.col(t).array()).matrix().cwiseProduct(dl);
    }

    *dx += this->Wxz_.transpose() * dzraw + this->Wxr_.transpose() * drraw +
      this->Wxh_.transpose() * dhuraw; // Backprop into x.
    *dx += Wxz_r_.transpose() * dzraw_r + Wxr_r_.transpose() * drraw_r +
      Wxl_.transpose() * dluraw; // Backprop into x.

    this->dWxz_.noalias() += dzraw * x.transpose();
    this->dbz_.noalias() += dzraw.rowwise().sum();
    this->dWxr_.noalias() += drraw * x.transpose();
    this->dbr_.noalias() += drraw.rowwise().sum();
    this->dWxh_.noalias() += dhuraw * x.transpose();
    this->dbh_.noalias() += dhuraw.rowwise().sum();

    dWxz_r_.noalias() += dzraw_r * x.transpose();
    dbz_r_.noalias() += dzraw_r.rowwise().sum();
    dWxr_r_.noalias() += drraw_r * x.transpose();
    dbr_r_.noalias() += drraw_r.rowwise().sum();
    dWxl_.noalias() += dluraw * x.transpose();
    dbl_.noalias() += dluraw.rowwise().sum();

    this->dWhz_.noalias() += dzraw.rightCols(length-1) *
      this->GetMutableOutput()->block(0, 0, this->hidden_size_, length-1).transpose();
    this->dWhr_.noalias() += drraw.rightCols(length-1) *
      this->GetMutableOutput()->block(0, 0, this->hidden_size_, length-1).transpose();
    this->dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((this->r_.rightCols(length-1)).cwiseProduct(this->GetMutableOutput()->block(0, 0, this->hidden_size_, length-1))).
      transpose();

    dWlz_r_.noalias() += dzraw_r.leftCols(length-1) *
      this->GetMutableOutput()->block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dWlr_r_.noalias() += drraw_r.leftCols(length-1) *
      this->GetMutableOutput()->block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dWll_.noalias() += dluraw.leftCols(length-1) *
      ((r_r_.leftCols(length-1)).cwiseProduct(this->GetMutableOutput()->block(this->hidden_size_, 1, this->hidden_size_, length-1))).
      transpose();

    this->dh0_.noalias() += dhnext;
    this->dl0_.noalias() += dlprev;

    //std::cout << "dl0=" << this->dl0_ << std::endl;
  }

 protected:
  Matrix<Real> Wxl_;
  Matrix<Real> Wll_;
  Matrix<Real> Wxz_r_;
  Matrix<Real> Wlz_r_;
  Matrix<Real> Wxr_r_;
  Matrix<Real> Wlr_r_;
  Vector<Real> bl_;
  Vector<Real> bz_r_;
  Vector<Real> br_r_;
  Vector<Real> l0_;

  Matrix<Real> dWxl_;
  Matrix<Real> dWll_;
  Matrix<Real> dWxz_r_;
  Matrix<Real> dWlz_r_;
  Matrix<Real> dWxr_r_;
  Matrix<Real> dWlr_r_;
  Vector<Real> dbl_;
  Vector<Real> dbz_r_;
  Vector<Real> dbr_r_;
  Vector<Real> dl0_;

  Matrix<Real> z_r_;
  Matrix<Real> r_r_;
  Matrix<Real> lu_;
};

#endif /* RNN_LAYER_H_ */
