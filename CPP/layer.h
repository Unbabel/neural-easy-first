#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "parameters.h"

template<typename Real> class Layer {
 public:
  Layer() {}
  ~Layer() {
    for (int i = 0; i < first_bias_moments_.size(); ++i) {
      delete first_bias_moments_[i];
      delete second_bias_moments_[i];
    }
    for (int i = 0; i < first_weight_moments_.size(); ++i) {
      delete first_weight_moments_[i];
      delete second_weight_moments_[i];
    }
  }

  const std::string &name() { return name_; }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) = 0;

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) = 0;

  virtual void UpdateParameters(int batch_size,
                                double learning_rate,
                                double regularization_constant) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];
      (*db) /= static_cast<double>(batch_size);
      // Note: the biases are not regularized.
      //(*db) += (regularization_constant * (*b));
      *b -= learning_rate * (*db);
    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];
      (*dW) /= static_cast<double>(batch_size);
      (*dW) += (regularization_constant * (*W));
      *W -= learning_rate * (*dW);
    }
  }

  double ComputeSquaredNormOfParameters() {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    // The biases do not contribute to the squared norm.
    double squared_norm = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      squared_norm += (W->array() * W->array()).sum();
    }

    return squared_norm;
  }

  void UpdateParametersADAM(int batch_size,
                            double learning_rate,
                            double regularization_constant) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    bool adagrad = false;
    double stepsize = learning_rate *
      sqrt(1.0 - pow(beta2_, iteration_number_ + 1)) /
      (1.0 - pow(beta1_, iteration_number_ + 1));

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];

      (*db) /= static_cast<double>(batch_size);
      // Note: the biases are not regularized.
      // (*db) += (regularization_constant * (*b));

      auto mb = first_bias_moments_[i];
      auto vb = second_bias_moments_[i];

      if (adagrad) {
        *vb = vb->array() + (db->array() * db->array());
        *b = b->array() - learning_rate * db->array() /
          (epsilon_ + vb->array().sqrt());
      } else {
        *mb = beta1_ * (*mb) + (1.0 - beta1_) * (*db);
        *vb = beta2_ * vb->array() +
          (1.0 - beta2_) * (db->array() * db->array());
        *b = b->array() - stepsize * mb->array()
          / (epsilon_ + vb->array().sqrt());
      }
    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];

      (*dW) /= static_cast<double>(batch_size);
      (*dW) += (regularization_constant * (*W));

      auto mW = first_weight_moments_[i];
      auto vW = second_weight_moments_[i];

      if (adagrad) {
        *vW = vW->array() + (dW->array() * dW->array());
        *W = W->array() - learning_rate * dW->array() /
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

  void LoadParameters(const std::string &prefix, bool binary_mode) {
    std::vector<Matrix<Real>*> weights;
    std::vector<Vector<Real>*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto name = bias_names[i];
      if (binary_mode) {
        LoadVectorParameterFromFile(prefix, name, b);
      } else {
        std::cout << "Loading " << name << "..." << std::endl;
        ReadVectorParameter(prefix, name, b);
      }
    }
    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto name = weight_names[i];
      if (binary_mode) {
        LoadMatrixParameterFromFile(prefix, name, W);
      } else {
        std::cout << "Loading " << name << "..." << std::endl;
        ReadMatrixParameter(prefix, name, W);
      }
    }
  }

  void SaveParameters(const std::string &prefix, bool binary_mode) {
    std::vector<Matrix<Real>*> weights;
    std::vector<Vector<Real>*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto name = bias_names[i];
      if (binary_mode) {
        SaveVectorParameterToFile(prefix, name, b);
      } else {
        // TODO: Implement this function.
        assert(false);
        //WriteVectorParameter(prefix, name, b);
      }
    }
    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto name = weight_names[i];
      if (binary_mode) {
        SaveMatrixParameterToFile(prefix, name, W);
      } else {
        // TODO: Implement this function.
        assert(false);
        //WriteMatrixParameter(prefix, name, W);
      }
    }
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

  void CheckGradient(int num_checks, double delta) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    //ResetGradients(); NOTE: this needs to be called by the test function!!!
    RunForward();
    RunBackward();

    std::cout << name_;
    for (int k = 0; k < GetNumOutputs(); ++k) {
      std::cout << " " << GetMutableOutputDerivative(k)->size();
      assert(GetOutput(k).size() == GetMutableOutputDerivative(k)->size());
    }
    std::cout << std::endl;

    for (int check = 0; check < num_checks; ++check) {
      for (int i = 0; i < biases.size(); ++i) {
        auto name = bias_names[i];
        auto b = biases[i];
        auto db = bias_derivatives[i];
        int r = static_cast<int>(b->size() *
                                 static_cast<double>(rand()) / RAND_MAX);
        double value = (*b)[r];
        (*b)[r] = value + delta;
        RunForward();
        double out0 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out0 += (GetOutput(k).array() * GetMutableOutputDerivative(k)->array()).sum();
        }
        (*b)(r) = value - delta;
        RunForward();
        double out1 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out1 += (GetOutput(k).array() * GetMutableOutputDerivative(k)->array()).sum();
        }
        (*b)(r) = value; // Put the value back.
        RunForward();
        double numeric_gradient = (out0 - out1) / (2 * delta);
        double analytic_gradient = (*db)[r];
        double relative_error = 0.0;
        if (numeric_gradient + analytic_gradient != 0.0) {
          relative_error = fabs(numeric_gradient - analytic_gradient) /
            fabs(numeric_gradient + analytic_gradient);
        }
        std::cout << name << ": "
                  << numeric_gradient << ", " << analytic_gradient
                  << " => " << relative_error
                  << std::endl;
      }
      for (int i = 0; i < weights.size(); ++i) {
        auto name = weight_names[i];
        auto W = weights[i];
        auto dW = weight_derivatives[i];
        int r = static_cast<int>(W->size() *
                                 static_cast<double>(rand()) / RAND_MAX);
        double value = (*W)(r);
        (*W)(r) = value + delta;
        RunForward();
        double out0 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out0 += (GetOutput(k).array() * GetMutableOutputDerivative(k)->array()).sum();
        }
        (*W)(r) = value - delta;
        RunForward();
        double out1 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out1 += (GetOutput(k).array() * GetMutableOutputDerivative(k)->array()).sum();
        }
        (*W)(r) = value; // Put the value back.
        RunForward();
        double numeric_gradient = (out0 - out1) / (2 * delta);
        double analytic_gradient = (*dW)(r);
        double relative_error = 0.0;
        if (numeric_gradient + analytic_gradient != 0.0) {
          relative_error = fabs(numeric_gradient - analytic_gradient) /
            fabs(numeric_gradient + analytic_gradient);
        }
        std::cout << name << "(" << r << "/" << W->size() << ")" << ": "
                  << numeric_gradient << ", " << analytic_gradient
                  << " => " << relative_error
                  << std::endl;
      }
    }

    std::cout << std::endl;
  }

  virtual void ResetOutputDerivatives() {
    for (int k = 0; k < GetNumOutputs(); ++k) {
      output_derivatives_[k].setZero(outputs_[k].rows(), outputs_[k].cols());
    }
  }

  virtual void RunForward() = 0;
  virtual void RunBackward() = 0;
  //virtual void UpdateParameters(double learning_rate) = 0;

  void ApplyDropout(int k, double dropout_probability) {
    for (int i = 0; i < outputs_[k].rows(); ++i) {
      for (int j = 0; j < outputs_[k].cols(); ++j) {
        double value =  static_cast<double>(rand()) / RAND_MAX;
        if (value < dropout_probability) {
          outputs_[k](i,j) = 0.0;
        }
      }
    }
  }

  void ScaleOutput(int k, double scale_factor) {
    outputs_[k] *= scale_factor;
  }

  int GetNumInputs() const { return inputs_.size(); }
  void SetNumInputs(int n) {
    inputs_.resize(n);
    input_derivatives_.resize(n);
  }
  void SetInput(int i, const Matrix<Real> &input) { inputs_[i] = &input; }
  void SetInputDerivative(int i, Matrix<Real> *input_derivative) {
    input_derivatives_[i] = input_derivative;
  }

  int GetNumOutputs() const { return outputs_.size(); }
  void SetNumOutputs(int n) {
    outputs_.resize(n);
    output_derivatives_.resize(n);
  }
  const Matrix<Real> &GetOutput(int i) const { return outputs_[i]; }
  Matrix<Real> *GetMutableOutputDerivative(int i) { return &(output_derivatives_[i]); }

 protected:
  void SetOutput(int k, const Matrix<Real> &output) {
    outputs_[k] = output;
  }

  // Methods that assume single input/output.
  const Matrix<Real> &GetInput() {
    assert(GetNumInputs() == 1);
    return *inputs_[0];
  }
  Matrix<Real> *GetInputDerivative() {
    assert(GetNumInputs() == 1);
    return input_derivatives_[0];
  }
  Matrix<Real> *GetMutableOutput() {
    //std::cout << name_ << std::endl;
    assert(GetNumOutputs() == 1);
    return &(outputs_[0]);
  }
  void SetOutput(const Matrix<Real> &output) {
    assert(GetNumOutputs() == 1);
    outputs_[0] = output;
  }
  const Matrix<Real> &GetOutputDerivative() {
    assert(GetNumOutputs() == 1);
    return output_derivatives_[0];
  }

 protected:
  std::string name_;
  std::vector<const Matrix<Real>*> inputs_;
  std::vector<Matrix<Real> > outputs_;
  std::vector<Matrix<Real>*> input_derivatives_;
  std::vector<Matrix<Real> > output_derivatives_;

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

#endif /* LAYER_H_ */
