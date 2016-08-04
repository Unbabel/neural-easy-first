#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"

template<typename Real> class Parameters {
 public:
  Parameters() {}
  ~Parameters() { DeleteAllParameters(); }

  void DeleteAllParameters() {
    for (auto W: weights_) {
      delete W;
    }
    for (auto dW: weight_derivatives_) {
      delete dW;
    }
    for (auto b: biases_) {
      delete b;
    }
    for (auto db: bias_derivatives_) {
      delete db;
    }
    weights_.clear();
    biases_.clear();
    weight_derivatives_.clear();
    bias_derivatives_.clear();
    weight_names_.clear();
    bias_names_.clear();
  }

  const std::vector<Matrix<Real>*> &weights() { return weights_; }
  const std::vector<Matrix<Real>*> &weight_derivatives() {
    return weight_derivatives_;
  }
  const std::vector<Vector<Real>*> &biases() { return biases_; }
  const std::vector<Vector<Real>*> &bias_derivatives() {
    return bias_derivatives_;
  }
  const std::vector<std::string> &weight_names() { return weight_names_; }
  const std::vector<std::string> &bias_names() { return bias_names_; }

  void CreateMatrixParameter(const std::string &name,
                             int num_rows,
                             int num_columns,
                             Matrix<Real> **W,
                             Matrix<Real> **dW) {
    // Create a matrix of weights and another one for derivatives of those
    // weights.
    Matrix<Real>* parameter = new Matrix<Real>(num_rows, num_columns);
    parameter->setZero();
    weights_.push_back(parameter);
    Matrix<Real>* gradient = new Matrix<Real>(num_rows, num_columns);
    gradient->setZero();
    weight_derivatives_.push_back(gradient);
    weight_names_.push_back(name);
    *W = parameter;
    *dW = gradient;
  }

  void CreateVectorParameter(const std::string &name,
                             int num_elements,
                             Vector<Real> **b,
                             Vector<Real> **db) {
    // Create a vector of weights and another one for derivatives of those
    // weights.
    Vector<Real>* parameter = new Vector<Real>(num_elements);
    parameter->setZero();
    biases_.push_back(parameter);
    Vector<Real>* gradient = new Vector<Real>(num_elements);
    gradient->setZero();
    bias_derivatives_.push_back(gradient);
    bias_names_.push_back(name);
    *b = parameter;
    *db = gradient;
  }

  double GetUniformInitializationLimit(Matrix<Real> *W,
                                       int activation_function) {
    // Do Glorot initialization.
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    if (activation_function == ActivationFunctions::LOGISTIC) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    for (int i = 0; i < weights_.size(); ++i) {
      weight_derivatives_.setZero(weights_.rows(), weights_.cols());
    }
    for (int i = 0; i < biases_.size(); ++i) {
      bias_derivatives_.setZero(biases_.rows(), biases_.cols());
    }
  }

  double ComputeSquaredNormOfParameters() {
    // The biases do not contribute to the squared norm.
    double squared_norm = 0.0;
    for (int i = 0; i < weights_.size(); ++i) {
      auto W = weights_[i];
      squared_norm += (W->array() * W->array()).sum();
    }

    return squared_norm;
  }

  void InitializeParameters() {
    for (auto b: biases_) {
      b->setZero();
    }
    for (auto W: weights_) {
      double max = GetUniformInitializationLimit(W);
      for (int i = 0; i < W->rows(); ++i) {
        for (int j = 0; j < W->cols(); ++j) {
          double t = max *
            (2.0*static_cast<double>(rand()) / RAND_MAX - 1.0);
          (*W)(i, j) = t;
          //std::cout << t/max << std::endl;
        }
      }
    }
  }

  void LoadParameters(const std::string &prefix, bool binary_mode) {
    for (int i = 0; i < biases_.size(); ++i) {
      auto b = biases_[i];
      auto name = bias_names_[i];
      if (binary_mode) {
        LoadVectorParameterFromFile(prefix, name, b);
      } else {
        std::cout << "Loading " << name << "..." << std::endl;
        ReadVectorParameter(prefix, name, b);
      }
    }
    for (int i = 0; i < weights_.size(); ++i) {
      auto W = weights_[i];
      auto name = weight_names_[i];
      if (binary_mode) {
        LoadMatrixParameterFromFile(prefix, name, W);
      } else {
        std::cout << "Loading " << name << "..." << std::endl;
        ReadMatrixParameter(prefix, name, W);
      }
    }
  }

  void SaveParameters(const std::string &prefix, bool binary_mode) {
    for (int i = 0; i < biases_.size(); ++i) {
      auto b = biases_[i];
      auto name = bias_names_[i];
      if (binary_mode) {
        SaveVectorParameterToFile(prefix, name, b);
      } else {
        // TODO: Implement this function.
        assert(false);
        //WriteVectorParameter(prefix, name, b);
      }
    }
    for (int i = 0; i < weights_.size(); ++i) {
      auto W = weights_[i];
      auto name = weight_names_[i];
      if (binary_mode) {
        SaveMatrixParameterToFile(prefix, name, W);
      } else {
        // TODO: Implement this function.
        assert(false);
        //WriteMatrixParameter(prefix, name, W);
      }
    }
  }

 protected:
  std::vector<Matrix<Real>*> weights_;
  std::vector<Matrix<Real>*> weight_derivatives_;
  std::vector<Vector<Real>*> biases_;
  std::vector<Vector<Real>*> bias_derivatives_;
  std::vector<std::string> weight_names_;
  std::vector<std::string> bias_names_;
};

#endif /* PARAMETERS_H_ */
