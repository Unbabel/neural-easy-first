#ifndef EASY_FIRST_LAYER_H_
#define EASY_FIRST_LAYER_H_

#include "layer.h"

template<typename Real> class EasyFirstLayer : public Layer<Real> {
 public:
  EasyFirstLayer() {}
  EasyFirstLayer(int input_size,
                 int hidden_size,
                 int context_size,
                 int attention_type) {
    this->name_ = "EasyFirst";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    // Size of context for the convolutional layer.
    context_size_ = context_size;
    attention_type_ = attention_type;
    // TODO: Set params to NULL.
  }
  virtual ~EasyFirstLayer() { DeleteParameters(); }

  void DeleteParameters() {
    // TODO.
  }

  void RunForward() {
    const Matrix<Real> &X = this->GetInput();
    int length = X.cols();
    // Sketch matrix.
    Matrix<Real> S = Matrix<Real>::Zeros(hidden_size_, length);
    for (int i = 0; i < num_steps_; ++i) {
      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetNumOutputs(1);
      concatenator_layer_->SetInput(0, X);
      concatenator_layer_->SetInput(1, S);

      convolutional_layer_->SetContextSize(context_size);
      convolutional_layer_->SetNumInputs(1);
      convolutional_layer_->SetNumOutputs(1);
      convolutional_layer_->SetInput(0, concatenator_layer_->GetOutput(0));

      // The input is already the concatenation of the input with the current
      // sketch.
      attention_layer_->SetNumInputs(1);
      // The outputs are the attention probabilities and the weighted average.
      attention_layer_->SetNumOutputs(2);
      attention_layer_->SetInput(0, convolutional_layer_->GetOutput(0));

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetNumOutputs(1);
      feedforward_layer_->SetInput(0, attention_layer_->GetOutput(1));

      const Vector<Real> &sketch_vector = feedforward_layer_->GetOutput(0);
      const Vector<Real> &attention_probabilities = attention_->GetOutput(0);

      S += sketch_vector * attention_probabilities.transpose();
    }

    // TODO.
  }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  int context_size_;
  int attention_type_;
  int num_steps_;
  AttentionLayer<Real> *attention_layer_;
  ConvolutionalLayer<Real> *convolutional_layer_;
  ConcatenatorLayer<Real> *concatenator_layer_;
  FeedforwardLayer<Real> *feedforward_layer_;
};

#endif /* EASY_FIRST_LAYER_H_ */
