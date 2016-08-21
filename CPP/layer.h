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
  ~Layer() {}

  const std::string &name() { return name_; }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) = 0;

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) = 0;

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
          out0 += (GetOutput(k).array() *
                   GetMutableOutputDerivative(k)->array()).sum();
        }
        (*b)(r) = value - delta;
        RunForward();
        double out1 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out1 += (GetOutput(k).array() *
                   GetMutableOutputDerivative(k)->array()).sum();
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
          out0 += (GetOutput(k).array() *
                   GetMutableOutputDerivative(k)->array()).sum();
        }
        (*W)(r) = value - delta;
        RunForward();
        double out1 = 0.0;
        for (int k = 0; k < GetNumOutputs(); ++k) {
          out1 += (GetOutput(k).array() *
                   GetMutableOutputDerivative(k)->array()).sum();
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
};

#endif /* LAYER_H_ */
