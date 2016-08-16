#ifndef UPDATER_H_
#define UPDATER_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "parameters.h"

template<typename Real> class Updater {
 public:
  Updater() {
    parameters_ = NULL;
    l2_regularization_constant_ = 0.0;
  }
  Updater(Parameters<Real> *parameters) : parameters_(parameters) {
    l2_regularization_constant_ = 0.0;
  }
  virtual ~Updater() {}

  void set_l2_regularization_constant(double l2_regularization_constant) {
    l2_regularization_constant_ = l2_regularization_constant;
  }

  virtual void UpdateParameters(int batch_size) = 0;

 protected:
  double l2_regularization_constant_;
  Parameters<Real> *parameters_;
};

template<typename Real> class SGDUpdater : public Updater<Real> {
 public:
  SGDUpdater(Parameters<Real> *parameters) {
    this->parameters_ = parameters;
  }
  virtual ~SGDUpdater() {}

  void set_learning_rate(double learning_rate) {
    learning_rate_ = learning_rate;
  }

  void UpdateParameters(int batch_size) {
    for (int i = 0; i < this->parameters_->biases().size(); ++i) {
      auto b = this->parameters_->biases()[i];
      auto db = this->parameters_->bias_derivatives()[i];
      (*db) /= static_cast<double>(batch_size);
      // Note: the biases are not regularized.
      //(*db) += (regularization_constant * (*b));
      *b -= learning_rate_ * (*db);
    }

    for (int i = 0; i < this->parameters_->weights().size(); ++i) {
      auto W = this->parameters_->weights()[i];
      auto dW = this->parameters_->weight_derivatives()[i];
      (*dW) /= static_cast<double>(batch_size);
      (*dW) += (this->l2_regularization_constant_ * (*W));
      *W -= learning_rate_ * (*dW);
    }
  }

 protected:
  double learning_rate_;
};

template<typename Real> class ADAMUpdater : public Updater<Real> {
 public:
  ADAMUpdater(Parameters<Real> *parameters) {
    this->parameters_ = parameters;
    Initialize(0.9, 0.999, 1e-8);
  }
  virtual ~ADAMUpdater() {}

  void Initialize(double beta1, double beta2, double epsilon) {
    beta1_ = beta1;
    beta2_ = beta2;
    epsilon_ = epsilon;
    iteration_number_ = 0;

    first_bias_moments_.resize(this->parameters_->biases().size());
    second_bias_moments_.resize(this->parameters_->biases().size());
    for (int i = 0; i < this->parameters_->biases().size(); ++i) {
      auto b = this->parameters_->biases()[i];
      first_bias_moments_[i] = new Vector<Real>;
      first_bias_moments_[i]->setZero(b->size());
      second_bias_moments_[i] = new Vector<Real>;
      second_bias_moments_[i]->setZero(b->size());
    }

    first_weight_moments_.resize(this->parameters_->weights().size());
    second_weight_moments_.resize(this->parameters_->weights().size());
    for (int i = 0; i < this->parameters_->weights().size(); ++i) {
      auto W = this->parameters_->weights()[i];
      first_weight_moments_[i] = new Matrix<Real>;
      first_weight_moments_[i]->setZero(W->rows(), W->cols());
      second_weight_moments_[i] = new Matrix<Real>;
      second_weight_moments_[i]->setZero(W->rows(), W->cols());
    }
  }

  void set_learning_rate(double learning_rate) {
    learning_rate_ = learning_rate;
  }

  void UpdateParameters(int batch_size) {
    bool adagrad = false;
    double stepsize = learning_rate_ *
      sqrt(1.0 - pow(beta2_, iteration_number_ + 1)) /
      (1.0 - pow(beta1_, iteration_number_ + 1));

    for (int i = 0; i < this->parameters_->biases().size(); ++i) {
      auto b = this->parameters_->biases()[i];
      auto db = this->parameters_->bias_derivatives()[i];

      (*db) /= static_cast<double>(batch_size);
      // Note: the biases are not regularized.
      // (*db) += (regularization_constant * (*b));

      auto mb = first_bias_moments_[i];
      auto vb = second_bias_moments_[i];

      if (adagrad) {
        *vb = vb->array() + (db->array() * db->array());
        *b = b->array() - learning_rate_ * db->array() /
          (epsilon_ + vb->array().sqrt());
      } else {
        *mb = beta1_ * (*mb) + (1.0 - beta1_) * (*db);
        *vb = beta2_ * vb->array() +
          (1.0 - beta2_) * (db->array() * db->array());
        *b = b->array() - stepsize * mb->array()
          / (epsilon_ + vb->array().sqrt());
      }
    }

    for (int i = 0; i < this->parameters_->weights().size(); ++i) {
      auto W = this->parameters_->weights()[i];
      auto dW = this->parameters_->weight_derivatives()[i];

      (*dW) /= static_cast<double>(batch_size);
      (*dW) += (this->l2_regularization_constant_ * (*W));

      auto mW = first_weight_moments_[i];
      auto vW = second_weight_moments_[i];

      if (adagrad) {
        *vW = vW->array() + (dW->array() * dW->array());
        *W = W->array() - learning_rate_ * dW->array() /
          (epsilon_ + vW->array().sqrt());
      } else {
        *mW = beta1_ * (*mW) + (1.0 - beta1_) * (*dW);
        *vW = beta2_ * vW->array() +
          (1.0 - beta2_) * (dW->array() * dW->array());
        *W = W->array() - stepsize * mW->array()
          / (epsilon_ + vW->array().sqrt());
      }
    }

    ++iteration_number_;
  }

  void LoadADAMParameters(const std::string &prefix) {
    std::string param_file = prefix + "ADAM" + ".bin";
    FILE *fs = fopen(param_file.c_str(), "rb");
    if (1 != fread(&iteration_number_, sizeof(int), 1, fs)) assert(false);
    double value;
    if (1 != fread(&value, sizeof(double), 1, fs)) assert(false);
    beta1_ = value;
    if (1 != fread(&value, sizeof(double), 1, fs)) assert(false);
    beta2_ = value;
    if (1 != fread(&value, sizeof(double), 1, fs)) assert(false);
    epsilon_ = value;
    int length;
    if (1 != fread(&length, sizeof(int), 1, fs)) assert(false);
    first_weight_moments_.resize(length);
    for (int k = 0; k < length; ++k) {
      LoadMatrixParameter(fs, first_weight_moments_[k]);
    }
    if (1 != fread(&length, sizeof(int), 1, fs)) assert(false);
    first_bias_moments_.resize(length);
    for (int k = 0; k < length; ++k) {
      LoadVectorParameter(fs, first_bias_moments_[k]);
    }
    if (1 != fread(&length, sizeof(int), 1, fs)) assert(false);
    second_weight_moments_.resize(length);
    for (int k = 0; k < length; ++k) {
      LoadMatrixParameter(fs, second_weight_moments_[k]);
    }
    if (1 != fread(&length, sizeof(int), 1, fs)) assert(false);
    second_bias_moments_.resize(length);
    for (int k = 0; k < length; ++k) {
      LoadVectorParameter(fs, second_bias_moments_[k]);
    }
    fclose(fs);
  }

  void SaveADAMParameters(const std::string &prefix) {
    std::string param_file = prefix + "ADAM" + ".bin";
    FILE *fs = fopen(param_file.c_str(), "wb");
    if (1 != fwrite(&iteration_number_, sizeof(int), 1, fs)) assert(false);
    double value;
    value = beta1_;
    if (1 != fwrite(&value, sizeof(double), 1, fs)) assert(false);
    value = beta2_;
    if (1 != fwrite(&value, sizeof(double), 1, fs)) assert(false);
    value = epsilon_;
    if (1 != fwrite(&value, sizeof(double), 1, fs)) assert(false);
    int length;
    length = first_weight_moments_.size();
    if (1 != fwrite(&length, sizeof(int), 1, fs)) assert(false);
    for (int k = 0; k < length; ++k) {
      SaveMatrixParameter(fs, first_weight_moments_[k]);
    }
    length = first_bias_moments_.size();
    if (1 != fwrite(&length, sizeof(int), 1, fs)) assert(false);
    for (int k = 0; k < length; ++k) {
      SaveVectorParameter(fs, first_bias_moments_[k]);
    }
    length = second_weight_moments_.size();
    if (1 != fwrite(&length, sizeof(int), 1, fs)) assert(false);
    for (int k = 0; k < length; ++k) {
      SaveMatrixParameter(fs, second_weight_moments_[k]);
    }
    length = second_bias_moments_.size();
    if (1 != fwrite(&length, sizeof(int), 1, fs)) assert(false);
    for (int k = 0; k < length; ++k) {
      SaveVectorParameter(fs, second_bias_moments_[k]);
    }
    fclose(fs);
  }

 protected:
  double learning_rate_;
  // ADAM parameters.
  double beta1_;
  double beta2_;
  double epsilon_;
  int iteration_number_;
  std::vector<Matrix<Real>*> first_weight_moments_;
  std::vector<Vector<Real>*> first_bias_moments_;
  std::vector<Matrix<Real>*> second_weight_moments_;
  std::vector<Vector<Real>*> second_bias_moments_;
};

#endif /* UPDATER_H_ */
