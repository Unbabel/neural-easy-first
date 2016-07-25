#include "lookup_layer.h"
#include "linear_layer.h"
#include "rnn_layer.h"
#include "attention_layer.h"
#include "alignment_layer.h"
#include "concatenator_layer.h"
#include "feedforward_layer.h"
#include "softmax_layer.h"

int main(int argc, char** argv) {

  LinearLayer<double> linear_layer(5, 5);
  GRULayer<double> rnn_layer(5, 5);
  LSTMLayer<double> lstm_layer(5, 5);
  BiGRULayer<double> birnn_layer(5, 5);
  AttentionLayer<double> attention_layer(5, 5, 5, true);
  FeedforwardLayer<double> feedforward_layer(5, 5);
  SoftmaxOutputLayer<double> output_layer(5, 3);

  double delta = 1e-8; //1e-7;
  int num_checks = 50;
  int num_tokens = 4;

  DoubleMatrix x, dx;
  DoubleMatrix *dy, *dc;
  DoubleMatrix c0, dc0;

  srand(1234);

  linear_layer.InitializeParameters();
  linear_layer.ResetGradients();
  x = DoubleMatrix::Random(linear_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(linear_layer.input_size(), num_tokens);
  linear_layer.SetNumInputs(1);
  linear_layer.SetNumOutputs(1);
  linear_layer.SetInput(0, x);
  linear_layer.SetInputDerivative(0, &dx);
  dy = linear_layer.GetMutableOutputDerivative(0);
  dy->setRandom(linear_layer.output_size(), num_tokens);
  linear_layer.CheckGradient(num_checks, delta);

  feedforward_layer.InitializeParameters();
  feedforward_layer.ResetGradients();
  x = DoubleMatrix::Random(feedforward_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(feedforward_layer.input_size(), num_tokens);
  feedforward_layer.SetNumInputs(1);
  feedforward_layer.SetNumOutputs(1);
  feedforward_layer.SetInput(0, x);
  feedforward_layer.SetInputDerivative(0, &dx);
  dy = feedforward_layer.GetMutableOutputDerivative(0);
  dy->setRandom(feedforward_layer.output_size(), num_tokens);
  feedforward_layer.CheckGradient(num_checks, delta);

  rnn_layer.InitializeParameters();
  rnn_layer.ResetGradients();
  x = DoubleMatrix::Random(rnn_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(rnn_layer.input_size(), num_tokens);
  rnn_layer.SetNumInputs(1);
  rnn_layer.SetNumOutputs(1);
  rnn_layer.SetInput(0, x);
  rnn_layer.SetInputDerivative(0, &dx);
  dy = rnn_layer.GetMutableOutputDerivative(0);
  dy->setRandom(rnn_layer.hidden_size(), num_tokens);
  rnn_layer.CheckGradient(num_checks, delta);

  bool use_control = true;
  lstm_layer.InitializeParameters();
  lstm_layer.ResetGradients();
  x = DoubleMatrix::Random(lstm_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(lstm_layer.input_size(), num_tokens);
  if (use_control) {
    c0 = DoubleMatrix::Random(lstm_layer.input_size(), 1);
    dc0 = DoubleMatrix::Zero(lstm_layer.input_size(), 1);
    lstm_layer.set_use_control(true);
    lstm_layer.SetNumInputs(2);
    lstm_layer.SetNumOutputs(2);
    lstm_layer.SetInput(0, x);
    lstm_layer.SetInputDerivative(0, &dx);
    lstm_layer.SetInput(1, c0);
    lstm_layer.SetInputDerivative(1, &dc0);
  } else {
    lstm_layer.SetNumInputs(1);
    lstm_layer.SetNumOutputs(2);
    lstm_layer.SetInput(0, x);
    lstm_layer.SetInputDerivative(0, &dx);
  }
  dy = lstm_layer.GetMutableOutputDerivative(0);
  dy->setRandom(lstm_layer.hidden_size(), num_tokens);
  dc = lstm_layer.GetMutableOutputDerivative(1);
  dc->setRandom(lstm_layer.hidden_size(), num_tokens);
  lstm_layer.CheckGradient(num_checks, delta);

#if 0
  birnn_layer.InitializeParameters();
  birnn_layer.ResetGradients();
  x = DoubleMatrix::Random(birnn_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(birnn_layer.input_size(), num_tokens);
  birnn_layer.SetNumInputs(1);
  birnn_layer.SetNumOutputs(1);
  birnn_layer.SetInput(0, x);
  birnn_layer.SetInputDerivative(0, &dx);
  dy = birnn_layer.GetMutableOutputDerivative(0);
  dy->setRandom(2*birnn_layer.hidden_size(), num_tokens);
  birnn_layer.CheckGradient(num_checks, delta);
#endif

  attention_layer.InitializeParameters();
  attention_layer.ResetGradients();
  DoubleMatrix x1 = DoubleMatrix::Random(attention_layer.input_size(),
                                         num_tokens);
  DoubleMatrix x2 = DoubleMatrix::Random(attention_layer.control_size(), 1);
  DoubleMatrix dx1 = DoubleMatrix::Zero(attention_layer.input_size(),
                                        num_tokens);
  DoubleMatrix dx2 = DoubleMatrix::Zero(attention_layer.control_size(), 1);
  attention_layer.SetNumInputs(2);
  attention_layer.SetNumOutputs(1);
  attention_layer.SetInput(0, x1);
  attention_layer.SetInput(1, x2);
  attention_layer.SetInputDerivative(0, &dx1);
  attention_layer.SetInputDerivative(1, &dx2);
  dy = attention_layer.GetMutableOutputDerivative(0);
  dy->setRandom(attention_layer.hidden_size(), 1);
  attention_layer.CheckGradient(num_checks, delta);

  output_layer.InitializeParameters();
  output_layer.ResetGradients();
  x = DoubleMatrix::Random(output_layer.input_size(), 1);
  dx = DoubleMatrix::Zero(output_layer.input_size(), 1);
  output_layer.SetNumInputs(1);
  output_layer.SetNumOutputs(1);
  output_layer.SetInput(0, x);
  output_layer.SetInputDerivative(0, &dx);
  dy = output_layer.GetMutableOutputDerivative(0);
  //dy->setRandom(output_layer.output_size(), 1);
  output_layer.RunForward();
  int l = 0;
  output_layer.set_output_label(l);
  dy->setZero(output_layer.output_size(), 1);
  const DoubleMatrix &output = output_layer.GetOutput(0);
  (*dy)(l) = -1.0 / output(l);
  output_layer.CheckGradient(num_checks, delta);

#if 0
  Matrix *output_derivative = output_layer.GetMutableOutputDerivative(0);
  const Matrix &output = output_layer.GetOutput();
  output_derivative->setZero(output_layer.output_size(), 1);
  int l = output_layer.output_label();
  (*output_derivative)(l) = -1.0 / output(l);
  output_layer.CheckGradient(num_checks);

  attention_layer.CheckGradient(num_checks);
  feedforward_layer.CheckGradient(num_checks);
  rnn_layer.CheckGradient(num_checks);
#endif

  return 0;
}
