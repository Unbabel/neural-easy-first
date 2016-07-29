#ifndef ATTENTION_LAYER_H_
#define ATTENTION_LAYER_H_

#include "layer.h"

struct AttentionTypes {
  enum {
    LOGISTIC = 0,
    SOFTMAX,
    SPARSEMAX
  };
};

// This layer assumes:
// - An input X (input_size-by-N) whose columns are subject to attention.
// - Control variables Y (control_size-by-M) whose columns are controllers for
// the attention mechanisms.
// - The output will be a matrix Xtilde (input_size-by-M) whose columns are
// an weighted average of the rows in X determined by the M attention
// mechanisms.
// TODO: return also as output the probability matrix (N-by-M).
template<typename Real> class AttentionLayer : public Layer<Real> {
 public:
  AttentionLayer() {}
  AttentionLayer(int input_size,
                 int control_size,
                 int hidden_size,
                 int attention_type) {
    this->name_ = "Attention";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    control_size_ = control_size;
    hidden_size_ = hidden_size;
    attention_type_ = attention_type;
    Wxz_ = NULL;
    Wyz_ = NULL;
    wzp_ = NULL;
    bz_ = NULL;
  }
  virtual ~AttentionLayer() { DeleteParameters(); }

  int input_size() const { return input_size_; }
  int control_size() const { return control_size_; }
  int hidden_size() const { return hidden_size_; }

  void DeleteParameters() {
    delete Wxz_;
    delete Wyz_;
    delete wzp_;
    delete bz_;
  }

  void ResetParameters() {
    DeleteParameters();
    Wxz_ = new Matrix<Real>;
    Wxz_->setZero(hidden_size_, input_size_);
    Wyz_ = new Matrix<Real>;
    Wyz_->setZero(hidden_size_, control_size_);
    wzp_ = new Matrix<Real>;
    wzp_->setZero(hidden_size_, 1);
    bz_ = new Vector<Real>;
    bz_->setZero(hidden_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(Wxz_);
    weights->push_back(Wyz_);
    weights->push_back(wzp_);

    biases->push_back(bz_);

    weight_names->push_back("Wxz");
    weight_names->push_back("Wyz");
    weight_names->push_back("wzp");

    bias_names->push_back("bz");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWyz_);
    weight_derivatives->push_back(&dwzp_);
    bias_derivatives->push_back(&dbz_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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

  void ResetGradients() {
    dWxz_.setZero(hidden_size_, input_size_);
    dWyz_.setZero(hidden_size_, control_size_);
    dwzp_.setZero(hidden_size_, 1);
    dbz_.setZero(hidden_size_);
  }

  void RunForward() {
    assert(this->GetNumInputs() == 2);
    const Matrix<Real> &X = *(this->inputs_[0]); // Input subject to attention.
    const Matrix<Real> &Y = *(this->inputs_[1]); // Control inputs.
    int num_attention_mechanisms = Y.cols();
    int length = X.cols();

    // The current supported version of Eigen does not support tensors, so we
    // just represent a 3D tensor as a vector of Matrix objects.
    z_.resize(num_attention_mechanisms);
    P_.setZero(length, num_attention_mechanisms);
    for (int m = 0; m < num_attention_mechanisms; ++m) {
      z_[m].setZero(hidden_size_, length);
      Vector<Real> tmp, result;
      for (int t = 0; t < length; ++t) {
        tmp = *Wxz_ * X.col(t) + *bz_ + *Wyz_ * Y.col(m);
        EvaluateActivation(activation_function_,
                           tmp,
                           &result);
        z_[m].col(t) = result;
      }

      Vector<Real> v = z_[m].transpose() * *wzp_;
      if (attention_type_ == AttentionTypes::SPARSEMAX) {
        Real tau;
        Real r = 1.0;
        //std::cout << "v=" << v << std::endl;
        // TODO: Implement ProjectOntoSimplex with matrices to avoid the
        // overhead below.
        Vector<Real> p = P_.col(m);
        ProjectOntoSimplex(v, r, &p, &tau);
        P_.col(m) = p;
        //std::cout << "p=" << p_ << std::endl;
        //std::cout << "tau=" << tau << std::endl;
      } else if (attention_type_ == AttentionTypes::SOFTMAX) {
        Real logsum = LogSumExp(v);
        P_.col(m) = (v.array() - logsum).exp();
      } else { // LOGISTIC.
        Vector<Real> p = P_.col(m);
        EvaluateActivation(ActivationFunctions::LOGISTIC, v, &p);
        P_.col(m) = p;
      }
    }

    this->SetOutput(X * P_);
  }

  void RunBackward() {
    assert(this->GetNumInputs() == 2);
    const Matrix<Real> &X = *(this->inputs_[0]); // Input subject to attention.
    const Matrix<Real> &Y = *(this->inputs_[1]); // Control inputs.
    Matrix<Real> *dX = this->input_derivatives_[0];
    Matrix<Real> *dY = this->input_derivatives_[1];
    int num_attention_mechanisms = Y.cols();
    int length = X.cols();
    dP_.noalias() = X.transpose() * this->GetOutputDerivative();

    dz_.resize(num_attention_mechanisms);
    for (int m = 0; m < num_attention_mechanisms; ++m) {
      Vector<Real> Jdp;
      if (attention_type_ == AttentionTypes::SPARSEMAX) {
        // Compute Jparsemax * dp_.
        // Let s = supp(p_) and k = sum(s).
        // Jsparsemax = diag(s) - s*s.transpose() / k.
        // Jsparsemax * dp_ = s.array() * dp_.array() - s*s.transpose() * dp_ / k
        //                  = s.array() * dp_.array() - val * s,
        // where val = s.transpose() * dp_ / k.
        //
        // With array-indexing this would be:
        //
        // float val = dp_[mask].sum() / mask.size();
        // Jdp[mask] = dp_[mask] - val;
        int nnz = 0;
        float val = 0.0;
        for (int i = 0; i < P_.rows(); ++i) {
          if (P_(i, m) > 0.0) {
            val += dP_(i, m);
            ++nnz;
          }
        }
        val /= static_cast<float>(nnz);
        Jdp.setZero(P_.rows());
        for (int i = 0; i < P_.rows(); ++i) {
          if (P_(i, m) > 0.0) {
            Jdp[i] = dP_(i, m) - val;
          }
        }
      } else if (attention_type_ == AttentionTypes::SOFTMAX) {
        // Compute Jsoftmax * dp_.
        // Jsoftmax = diag(p_) - p_*p_.transpose().
        // Jsoftmax * dp_ = p_.array() * dp_.array() - p_* (p_.transpose() * dp_).
        //                = p_.array() * (dp_ - val).array(),
        // where val = p_.transpose() * dp_.
        float val = P_.col(m).transpose() * dP_.col(m);
        Jdp = P_.col(m).array() * (dP_.col(m).array() - val);
      } else { // // LOGISTIC.
        const Vector<Real> &p = P_.col(m);
        DerivateActivation(ActivationFunctions::LOGISTIC, p, &Jdp);
        Jdp = Jdp.array() * dP_.col(m).array();
      }
      dz_[m] = *wzp_ * Jdp.transpose();

      Matrix<Real> dzraw = Matrix<Real>::Zero(hidden_size_, length);
      for (int t = 0; t < length; ++t) {
        Vector<Real> result; // TODO: perform this in a matrix-level way to be more efficient.
        Vector<Real> tmp = z_[m].col(t);
        DerivateActivation(activation_function_, tmp, &result);
        dzraw.col(t).noalias() = result;
      }
      dzraw = dzraw.array() * dz_[m].array();
      Vector<Real> dzraw_sum = dzraw.rowwise().sum();

      *dX += Wxz_->transpose() * dzraw;
      // TODO: do this out of the loop like:
      // *dX += this->GetOutputDerivative() * P_.transpose();
      *dX += this->GetOutputDerivative().col(m) * P_.col(m).transpose();
      dY->col(m) += Wyz_->transpose() * dzraw_sum;

      dwzp_ += z_[m] * Jdp;
      dWxz_.noalias() += dzraw * X.transpose();
      dWyz_.noalias() += dzraw_sum * Y.col(m).transpose();
      dbz_.noalias() += dzraw_sum;
    }
  }

  const Matrix<Real> &GetAttentionProbabilities() {
    return P_;
  }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  int control_size_;
  int attention_type_;

  Matrix<Real> *Wxz_;
  Matrix<Real> *Wyz_;
  Matrix<Real> *wzp_; // Column vector.
  Vector<Real> *bz_;

  Matrix<Real> dWxz_;
  Matrix<Real> dWyz_;
  Matrix<Real> dwzp_;
  Vector<Real> dbz_;

  std::vector<Matrix<Real> > z_;
  Matrix<Real> P_;

  std::vector<Matrix<Real> > dz_;
  Matrix<Real> dP_;
};

#endif /* ATTENTION_LAYER_H_ */
