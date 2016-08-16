#ifndef ATTENTIVE_RNN_H_
#define ATTENTIVE_RNN_H_

#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "data.h"
#include "lookup_layer.h"
#include "linear_layer.h"
#include "rnn_layer.h"
#include "attention_layer.h"
#include "alignment_layer.h"
#include "concatenator_layer.h"
#include "feedforward_layer.h"
#include "softmax_layer.h"
#include "neural_network.h"
#include "updater.h"
#include "parameters.h"

class SentencePair {
 public:
  SentencePair() {}
  virtual ~SentencePair() {}

  const std::vector<Input> &source_sentence() const { return source_sentence_; }
  const std::vector<Input> &target_sentence() const { return target_sentence_; }
  const std::vector<std::pair<int, int> > &alignments() const {
    return alignments_;
  }

  void set_source_sentence(const std::vector<Input> &source_sentence) {
    source_sentence_ = source_sentence;
  }
  void set_target_sentence(const std::vector<Input> &target_sentence) {
    target_sentence_ = target_sentence;
  }
  void set_alignments(const std::vector<std::pair<int, int> > &alignments) {
    alignments_ = alignments;
  }

 protected:
  std::vector<Input> source_sentence_;
  std::vector<Input> target_sentence_;
  std::vector<std::pair<int, int> > alignments_;
};

template<typename Real> class AttentiveQualityEstimator :
    public NeuralNetwork<Real> {
 public:
  AttentiveQualityEstimator() {}
  AttentiveQualityEstimator(Dictionary *source_dictionary,
                            Dictionary *target_dictionary,
                            LabelAlphabet *label_alphabet,
                            int embedding_dimension,
                            int hidden_size,
                            int output_size,
                            double cost_false_positives,
                            double cost_false_negatives,
                            bool use_attention,
                            int attention_type) {
    source_dictionary_ = source_dictionary;
    target_dictionary_ = target_dictionary;
    label_alphabet_ = label_alphabet;
    write_attention_probabilities_ = false;
    use_ADAM_ = true; //false; //true;
    cost_augmented_ = true;
    cost_false_positives_ = cost_false_positives;
    cost_false_negatives_ = cost_false_negatives;
    use_attention_ = use_attention;
    attention_type_ = attention_type;
    use_lstms_ = false; //true; //false;
    use_bidirectional_rnns_ = true; //false;
    apply_dropout_ = false; // true; // false;
    dropout_probability_ = 0.1;
    test_ = false;
    input_size_ = hidden_size; // Size of the projected embedded words.
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    embedding_size_ = embedding_dimension;

    parameters_ = new Parameters<Real>();
    if (use_ADAM_) {
      updater_ = new ADAMUpdater<Real>(parameters_);
    } else {
      updater_ = new SGDUpdater<Real>(parameters_);
    }

    CreateNetwork();
  }

  virtual ~AttentiveQualityEstimator() {
    delete source_lookup_layer_;
    delete source_linear_layer_;
    delete target_lookup_layer_;
    delete target_linear_layer_;
    delete source_rnn_layer_;
    delete target_rnn_layer_;
    delete attention_layer_;
    delete alignment_layer_;
    delete concatenator_layer_;
    delete feedforward_layer_;
    delete output_layer_;
    delete parameters_;
    delete updater_;
  }

  void CreateNetwork() {
    // Random seed for parameter initialization.
    srand(1234);

    // Add source lookup and linear layers.
    source_lookup_layer_ =
      new LookupLayer<Real>(source_dictionary_->GetNumWords(), embedding_size_);
    source_lookup_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(source_lookup_layer_);
    source_linear_layer_ = new LinearLayer<Real>(embedding_size_, input_size_);
    source_linear_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(source_linear_layer_);

    // Add target lookup and linear layers.
    target_lookup_layer_ =
      new LookupLayer<Real>(target_dictionary_->GetNumWords(), embedding_size_);
    target_lookup_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(target_lookup_layer_);
    target_linear_layer_ = new LinearLayer<Real>(embedding_size_, input_size_);
    target_linear_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(target_linear_layer_);

    // Add source and target RNNs.
    int state_size;
    if (use_bidirectional_rnns_) {
      source_rnn_layer_ = new BiGRULayer<Real>(input_size_, hidden_size_);
      source_rnn_layer_->CreateParameters(parameters_);
      NeuralNetwork<Real>::AddLayer(source_rnn_layer_);
      target_rnn_layer_ = new BiGRULayer<Real>(input_size_, hidden_size_);
      target_rnn_layer_->CreateParameters(parameters_);
      NeuralNetwork<Real>::AddLayer(target_rnn_layer_);
      state_size = 2*hidden_size_;
    } else {
      source_rnn_layer_ = new GRULayer<Real>(input_size_, hidden_size_);
      source_rnn_layer_->CreateParameters(parameters_);
      NeuralNetwork<Real>::AddLayer(source_rnn_layer_);
      target_rnn_layer_ = new GRULayer<Real>(input_size_, hidden_size_);
      target_rnn_layer_->CreateParameters(parameters_);
      NeuralNetwork<Real>::AddLayer(target_rnn_layer_);
      state_size = hidden_size_;
    }

    if (use_attention_) {
      // Add an attention layer controlled by each target word.
      attention_layer_ = new AttentionLayer<Real>(state_size, state_size,
                                                  hidden_size_,
                                                  attention_type_);
      attention_layer_->CreateParameters(parameters_);
      NeuralNetwork<Real>::AddLayer(attention_layer_);
      alignment_layer_ = NULL;
    } else {
      // Add an alignment layer using pre-computed word alignments.
      // This layer has no parameters.
      alignment_layer_ = new AlignmentLayer<Real>();
      NeuralNetwork<Real>::AddLayer(alignment_layer_);
      attention_layer_ = NULL;
    }

    // Add a concatenator layer that concatenates the target states with the
    // source representation coming from the attention layer.
    // This layer has no parameters.
    concatenator_layer_ = new ConcatenatorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(concatenator_layer_);

    // Add a feedforward layer.
    feedforward_layer_ = new FeedforwardLayer<Real>(2*state_size,
                                                    hidden_size_);
    feedforward_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(feedforward_layer_);

    // Add a softmax layer that performs a softmax transformation for each
    // target position.
    output_layer_ = new SoftmaxLayer<Real>(hidden_size_, output_size_);
    output_layer_->CreateParameters(parameters_);
    NeuralNetwork<Real>::AddLayer(output_layer_);

    // Connect the layers.
    // Lookup/linear layers.
    source_lookup_layer_->SetNumInputs(1);
    source_lookup_layer_->SetNumOutputs(1);
    source_linear_layer_->SetNumInputs(1);
    source_linear_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(source_lookup_layer_,
                                       source_linear_layer_, 0, 0);
    target_lookup_layer_->SetNumInputs(1);
    target_lookup_layer_->SetNumOutputs(1);
    target_linear_layer_->SetNumInputs(1);
    target_linear_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(target_lookup_layer_,
                                       target_linear_layer_, 0, 0);

    // RNN layers.
    source_rnn_layer_->SetNumInputs(1);
    source_rnn_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(source_linear_layer_, source_rnn_layer_,
                                       0, 0);
    target_rnn_layer_->SetNumInputs(1);
    target_rnn_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(target_linear_layer_, target_rnn_layer_,
                                       0, 0);

    if (use_attention_) {
      // Attention layer.
      attention_layer_->SetNumInputs(2);
      attention_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(source_rnn_layer_, attention_layer_,
                                         0, 0);
      NeuralNetwork<Real>::ConnectLayers(target_rnn_layer_, attention_layer_,
                                         0, 1);
    } else {
      // Alignment layer.
      alignment_layer_->SetNumInputs(1);
      alignment_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(source_rnn_layer_, alignment_layer_,
                                         0, 0);
    }

    // Concatenator layers.
    concatenator_layer_->SetNumInputs(2);
    concatenator_layer_->SetNumOutputs(1);
    if (use_attention_) {
      NeuralNetwork<Real>::ConnectLayers(attention_layer_, concatenator_layer_,
                                         0, 0);
    } else {
      NeuralNetwork<Real>::ConnectLayers(alignment_layer_, concatenator_layer_,
                                         0, 0);
    }
    NeuralNetwork<Real>::ConnectLayers(target_rnn_layer_, concatenator_layer_,
                                       0, 1);

    // Feedforward layer.
    feedforward_layer_->SetNumInputs(1);
    feedforward_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(concatenator_layer_, feedforward_layer_,
                                       0, 0);

    // Softmax layer.
    output_layer_->SetNumInputs(1);
    output_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(feedforward_layer_, output_layer_, 0, 0);

    // Sort layers by topological order.
    NeuralNetwork<Real>::SortLayersByTopologicalOrder();
  }

  void SetModelPrefix(const std::string &model_prefix) {
    model_prefix_ = model_prefix;
  }

#if 0
  void InitializeParameters() {
    srand(1234);

    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->InitializeParameters();
    }

    if (use_ADAM_) {
      double beta1 = 0.9;
      double beta2 = 0.999;
      double epsilon = 1e-8;
      for (int k = 0; k < layers.size(); ++k) {
        layers[k]->InitializeADAM(beta1, beta2, epsilon);
      }
    }
  }
#endif

  void LoadModel(const std::string &prefix) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      std::ostringstream ss;
      ss << "Layer" << k << "_" << layers[k]->name() + "_";
      layers[k]->LoadParameters(prefix + ss.str(), true);
      if (use_ADAM_) {
        layers[k]->LoadADAMParameters(prefix + ss.str());
      }
    }
  }

  void SaveModel(const std::string &prefix) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      std::ostringstream ss;
      ss << "Layer" << k << "_" << layers[k]->name() + "_";
      layers[k]->SaveParameters(prefix + ss.str(), true);
      if (use_ADAM_) {
        layers[k]->SaveADAMParameters(prefix + ss.str());
      }
    }
  }

  void SetFixedEmbeddings(const Matrix<Real> &source_fixed_embeddings,
                          const Matrix<Real> &target_fixed_embeddings) {
    source_lookup_layer_->SetFixedEmbeddings(&source_fixed_embeddings);
    target_lookup_layer_->SetFixedEmbeddings(&target_fixed_embeddings);
  }

  void Evaluate(const std::vector<std::vector<int> > &all_gold_labels,
                const std::vector<std::vector<int> > &all_predicted_labels,
                double *accuracy,
                double *f1_bad,
                double *f1_ok) {
    int num_labels = label_alphabet_->GetNumLabels();
    int num_matched = 0;
    std::vector<int> num_matched_labels(num_labels, 0);
    std::vector<int> num_predicted_labels(num_labels, 0);
    std::vector<int> num_gold_labels(num_labels, 0);
    int num_words = 0;
    for (int i = 0; i < all_gold_labels.size(); ++i) {
      const std::vector<int> &predicted_labels = all_predicted_labels[i];
      int length = all_gold_labels[i].size();
      for (int j = 0; j < length; ++j) {
        int predicted_label = predicted_labels[j];
        int gold_label = all_gold_labels[i][j];
        if (gold_label == predicted_label) {
          num_matched_labels[predicted_label] += 1;
          num_matched += 1;
        }
        num_predicted_labels[predicted_label] += 1;
        num_gold_labels[gold_label] += 1;
      }
      num_words += length;
    }
    *accuracy = static_cast<double>(num_matched) / num_words;
    int bad = label_alphabet_->GetLabelId("BAD");
    double precision_bad = 0.0;
    double recall_bad = 0.0;
    *f1_bad = 0.0;
    if (num_matched_labels[bad] != 0) {
      precision_bad = static_cast<double>(num_matched_labels[bad]) /
        num_predicted_labels[bad];
      recall_bad = static_cast<double>(num_matched_labels[bad]) /
        num_gold_labels[bad];
      *f1_bad = 2*precision_bad*recall_bad / (precision_bad + recall_bad);
    }
    int ok = label_alphabet_->GetLabelId("OK");
    double precision_ok = 0.0;
    double recall_ok = 0.0;
    *f1_ok = 0.0;
    if (num_matched_labels[ok] != 0) {
      precision_ok = static_cast<double>(num_matched_labels[ok]) /
        num_predicted_labels[ok];
      recall_ok = static_cast<double>(num_matched_labels[ok]) /
        num_gold_labels[ok];
      *f1_ok = 2*precision_ok*recall_ok / (precision_ok + recall_ok);
    }
  }

  void Train(const std::vector<SentencePair> &sentence_pairs,
             const std::vector<std::vector<int> > &output_labels,
             const std::vector<SentencePair> &sentence_pairs_dev,
             const std::vector<std::vector<int> > &output_labels_dev,
             const std::vector<SentencePair> &sentence_pairs_test,
             const std::vector<std::vector<int> > &output_labels_test,
             int warm_start_on_epoch, // 0 for no warm-starting.
             int num_epochs,
             int batch_size,
             double learning_rate,
             double regularization_constant) {

    if (warm_start_on_epoch == 0) {
      // Initial performance.
      std::vector<std::vector<int> > predicted_labels_dev;
      int num_sentences_dev = sentence_pairs_dev.size();
      for (int i = 0; i < num_sentences_dev; ++i) {
        std::vector<int> predicted_labels;
        Run(sentence_pairs_dev[i], &predicted_labels);
        predicted_labels_dev.push_back(predicted_labels);
      }
      double accuracy_dev, f1_bad_dev, f1_ok_dev;
      Evaluate(output_labels_dev, predicted_labels_dev, &accuracy_dev,
               &f1_bad_dev, &f1_ok_dev);
      std::cout << " Initial accuracy dev: " << accuracy_dev
                << std::endl;
      std::cout << " Initial F1-OK dev: " << f1_ok_dev
                << std::endl;
      std::cout << " Initial F1-BAD dev: " << f1_bad_dev
                << std::endl;
      std::cout << " Initial F1-OK*F1-BAD dev: " << f1_ok_dev*f1_bad_dev
                << std::endl;

      SaveModel(model_prefix_ + "Epoch0_");
    } else {
      std::cout << "Warm-starting on epoch " << warm_start_on_epoch
                << "..." << std::endl;
      std::ostringstream ss;
      ss << "Epoch" << warm_start_on_epoch << "_";
      LoadModel(model_prefix_ + ss.str());
    }

    for (int epoch = warm_start_on_epoch; epoch < num_epochs; ++epoch) {
      std::ostringstream ss;
      TrainEpoch(sentence_pairs, output_labels,
                 sentence_pairs_dev, output_labels_dev,
                 sentence_pairs_test, output_labels_test,
                 epoch, batch_size, learning_rate, regularization_constant);
      // Uncomment this to save temporary model.
      //ss << "Epoch" << epoch+1 << "_";
      //SaveModel(model_prefix_ + ss.str());
    }

    SaveModel(model_prefix_);
  }

  void TrainEpoch(const std::vector<SentencePair> &sentence_pairs,
                  const std::vector<std::vector<int> > &output_labels,
                  const std::vector<SentencePair> &sentence_pairs_dev,
                  const std::vector<std::vector<int> > &output_labels_dev,
                  const std::vector<SentencePair> &sentence_pairs_test,
                  const std::vector<std::vector<int> > &output_labels_test,
                  int epoch,
                  int batch_size,
                  double learning_rate,
                  double regularization_constant) {
    timeval start, end;
    gettimeofday(&start, NULL);
    double total_loss = 0.0;
    //double accuracy = 0.0;
    int num_words = 0;
    int num_sentences = sentence_pairs.size();
    int actual_batch_size = 0;
    std::vector<std::vector<int> > all_predicted_labels;
    for (int i = 0; i < num_sentences; ++i) {
      if (i % batch_size == 0) {
        ResetParameterGradients();
        actual_batch_size = 0;
      }

      // If cost augmented, provide the costs to the output layer.
      int length = output_labels[i].size();
      int num_labels = label_alphabet_->GetNumLabels();
      int ok = label_alphabet_->GetLabelId("OK");
      output_layer_->set_cost_augmented(cost_augmented_);
      if (cost_augmented_) {
        Matrix<Real> costs = Matrix<Real>::Zero(num_labels, length);
        for (int j = 0; j < length; ++j) {
          for (int l = 0; l < num_labels; ++l) {
            if (l != output_labels[i][j]) {
              if (ok == output_labels[i][j]) {
                costs(l, j) = cost_false_positives_;
              } else {
                costs(l, j) = cost_false_negatives_;
              }
            }
          }
        }
        output_layer_->set_costs(costs);
      }
      RunForwardPass(sentence_pairs[i]);
      Matrix<Real> P = output_layer_->GetOutput(0);
      std::vector<int> prediction_labels;
      for (int j = 0; j < length; ++j) {
        double loss = -log(P(output_labels[i][j], j));
        int prediction;
        if (cost_augmented_) {
          const Matrix<Real> &P_original = output_layer_->GetProbabilities();
          P_original.col(j).maxCoeff(&prediction);
        } else {
          P.col(j).maxCoeff(&prediction);
        }
        prediction_labels.push_back(prediction);
        //if (prediction == output_labels[i][j]) {
        //  accuracy += 1.0;
        //}
        // TODO: Try with loss averaged over the sentence words.
        // This implies also averaging the gradient!
        total_loss += loss;
      }
      all_predicted_labels.push_back(prediction_labels);
      num_words += length;
      RunBackwardPass(sentence_pairs[i], output_labels[i], learning_rate,
                      regularization_constant);
      ++actual_batch_size;
      if (((i+1) % batch_size == 0) || (i == num_sentences-1)) {
        UpdateParameters(actual_batch_size, learning_rate,
                         regularization_constant);
      }
    }
    //accuracy /= num_words;
    total_loss /= num_words;
    double total_reg = 0.0;
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      total_reg += 0.5 * regularization_constant *
        layers[k]->ComputeSquaredNormOfParameters();
    }
    double accuracy, f1_bad, f1_ok;
    Evaluate(output_labels, all_predicted_labels, &accuracy, &f1_bad, &f1_ok);

    // At test time, we never perform cost-augmented decoding.
    output_layer_->set_cost_augmented(false);

    write_attention_probabilities_ = true;
    if (attention_type_ == AttentionTypes::SPARSEMAX) {
      os_attention_.open("sparse_attention.txt", std::ifstream::out);
    } else if (attention_type_ == AttentionTypes::SOFTMAX) {
      os_attention_.open("soft_attention.txt", std::ifstream::out);
    } else { // LOGISTIC.
      os_attention_.open("logistic_attention.txt", std::ifstream::out);
    }

    all_predicted_labels.clear();
    int num_sentences_dev = sentence_pairs_dev.size();
    int num_words_dev = 0;
    for (int i = 0; i < num_sentences_dev; ++i) {
      std::vector<int> predicted_labels;
      Run(sentence_pairs_dev[i], &predicted_labels);
      int length = output_labels_dev[i].size();
      all_predicted_labels.push_back(predicted_labels);
      num_words_dev += length;
    }
    double accuracy_dev, f1_bad_dev, f1_ok_dev;
    Evaluate(output_labels_dev, all_predicted_labels, &accuracy_dev,
             &f1_bad_dev, &f1_ok_dev);

    write_attention_probabilities_ = false;

    all_predicted_labels.clear();
    int num_sentences_test = sentence_pairs_test.size();
    int num_words_test = 0;
    for (int i = 0; i < num_sentences_test; ++i) {
      std::vector<int> predicted_labels;
      Run(sentence_pairs_test[i], &predicted_labels);
      int length = output_labels_test[i].size();
      all_predicted_labels.push_back(predicted_labels);
      num_words_test += length;
    }
    double accuracy_test, f1_bad_test, f1_ok_test;
    Evaluate(output_labels_test, all_predicted_labels, &accuracy_test,
             &f1_bad_test, &f1_ok_test);

    os_attention_.flush();
    os_attention_.clear();
    os_attention_.close();

    gettimeofday(&end, NULL);
    std::cout << "Epoch: " << epoch+1
              << " Total loss: " << total_loss
              << " Total reg: " << total_reg
              << " Total loss+reg: " << total_loss + total_reg
              << " Accuracy train: " << accuracy
              << " F1-OK train: " << f1_ok
              << " F1-BAD train: " << f1_bad
              << " F1-OK*F1-BAD train: " << f1_ok*f1_bad
              << " Accuracy dev: " << accuracy_dev
              << " F1-OK dev: " << f1_ok_dev
              << " F1-BAD dev: " << f1_bad_dev
              << " F1-OK*F1-BAD dev: " << f1_ok_dev*f1_bad_dev
              << " Accuracy test: " << accuracy_test
              << " F1-OK test: " << f1_ok_test
              << " F1-BAD test: " << f1_bad_test
              << " F1-OK*F1-BAD test: " << f1_ok_test*f1_bad_test
              << " Time: " << diff_ms(end,start)
              << std::endl;
  }

  void Test(const std::vector<SentencePair> &sentence_pairs_dev,
            const std::vector<std::vector<int> > &output_labels_dev,
            const std::vector<SentencePair> &sentence_pairs_test,
            const std::vector<std::vector<int> > &output_labels_test) {
    timeval start, end;
    gettimeofday(&start, NULL);

    write_attention_probabilities_ = true;
    if (attention_type_ == AttentionTypes::SPARSEMAX) {
      os_attention_.open("sparse_attention_test.txt", std::ifstream::out);
    } else if (attention_type_ == AttentionTypes::SOFTMAX) {
      os_attention_.open("soft_attention_test.txt", std::ifstream::out);
    } else { // LOGISTIC.
      os_attention_.open("logistic_attention_test.txt", std::ifstream::out);
    }

    std::vector<std::vector<int> > all_predicted_labels;
    int num_sentences_dev = sentence_pairs_dev.size();
    int num_words_dev = 0;
    for (int i = 0; i < num_sentences_dev; ++i) {
      std::vector<int> predicted_labels;
      Run(sentence_pairs_dev[i], &predicted_labels);
      int length = output_labels_dev[i].size();
      all_predicted_labels.push_back(predicted_labels);
      num_words_dev += length;
    }
    double accuracy_dev, f1_bad_dev, f1_ok_dev;
    Evaluate(output_labels_dev, all_predicted_labels, &accuracy_dev,
             &f1_bad_dev, &f1_ok_dev);

    write_attention_probabilities_ = false;

    all_predicted_labels.clear();
    int num_sentences_test = sentence_pairs_test.size();
    int num_words_test = 0;
    for (int i = 0; i < num_sentences_test; ++i) {
      std::vector<int> predicted_labels;
      Run(sentence_pairs_test[i], &predicted_labels);
      int length = output_labels_test[i].size();
      all_predicted_labels.push_back(predicted_labels);
      num_words_test += length;
    }
    double accuracy_test, f1_bad_test, f1_ok_test;
    Evaluate(output_labels_test, all_predicted_labels, &accuracy_test,
             &f1_bad_test, &f1_ok_test);

    os_attention_.flush();
    os_attention_.clear();
    os_attention_.close();

    gettimeofday(&end, NULL);
    std::cout << " Accuracy dev: " << accuracy_dev
              << " F1-OK dev: " << f1_ok_dev
              << " F1-BAD dev: " << f1_bad_dev
              << " F1-OK*F1-BAD dev: " << f1_ok_dev*f1_bad_dev
              << " Accuracy test: " << accuracy_test
              << " F1-OK test: " << f1_ok_test
              << " F1-BAD test: " << f1_bad_test
              << " F1-OK*F1-BAD test: " << f1_ok_test*f1_bad_test
              << " Time: " << diff_ms(end,start)
              << std::endl;
  }

  void Run(const SentencePair &sentence_pair,
           std::vector<int> *predicted_labels) {
    bool apply_dropout = apply_dropout_;
    test_ = true;
    apply_dropout_ = false; // TODO: Remove this line to have correct dropout at test time.
    RunForwardPass(sentence_pair);
    test_ = false;
    apply_dropout_ = apply_dropout;
    Matrix<Real> P = output_layer_->GetOutput(0);
    std::vector<int> predictions(P.cols());
    for (int j = 0; j < P.cols(); ++j) {
      P.col(j).maxCoeff(&predictions[j]);
    }
    *predicted_labels = predictions;
  }

  void RunForwardPass(const SentencePair &sentence_pair) {
    const std::vector<Input> &source_sentence = sentence_pair.source_sentence();
    const std::vector<Input> &target_sentence = sentence_pair.target_sentence();
    std::vector<int> source_words;
    for (auto input: source_sentence) {
      source_words.push_back(input.wid());
    }
    std::vector<int> target_words;
    for (auto input: target_sentence) {
      target_words.push_back(input.wid());
    }
    source_lookup_layer_->set_input_sequence(source_words);
    target_lookup_layer_->set_input_sequence(target_words);

    if (!use_attention_) {
      Matrix<Real> alignment_matrix =
        Matrix<Real>::Zero(source_sentence.size(), target_sentence.size());
      const std::vector<std::pair<int, int> > &alignments =
        sentence_pair.alignments();
      for (auto alignment: alignments) {
        int s = alignment.first;
        int t = alignment.second;
        alignment_matrix(s, t) = 1.0;
      }
      for (int t = 0; t < alignment_matrix.cols(); ++t) {
        double sum = alignment_matrix.col(t).sum();
        if (sum > 0.0) {
          alignment_matrix.col(t) /= sum;
        } else {
          alignment_matrix.col(t).array() +=
            1.0 / static_cast<double>(source_sentence.size());
        }
      }
      alignment_layer_->SetAlignmentMatrix(alignment_matrix);
    }

    //int state_size;
    //if (use_bidirectional_rnns_) {
    //  state_size = 2*hidden_size_;
    //} else {
    //  state_size = hidden_size_;
    //}

    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      //std::cout << "Begin forward " << layers[k]->name() << std::endl;
      layers[k]->RunForward();
      //std::cout << "End forward " << layers[k]->name() << std::endl;
      if (apply_dropout_) {
        if (layers[k] == source_lookup_layer_ ||
            layers[k] == target_lookup_layer_) {
          if (test_) {
            layers[k]->ScaleOutput(0, 1.0 - dropout_probability_);
          } else {
            layers[k]->ApplyDropout(0, dropout_probability_);
          }
        } else if (layers[k] == target_rnn_layer_) { // TODO: also source?
          if (test_) {
            target_rnn_layer_->ScaleOutput(0,
                                           1.0 - dropout_probability_);
          } else {
            target_rnn_layer_->ApplyDropout(0, dropout_probability_);
          }
        }
      }
    }

    if (use_attention_ && write_attention_probabilities_) {
      // Do something.
    }
  }

  void RunBackwardPass(const SentencePair &sentence_pair,
                       const std::vector<int> &output_labels,
                       double learning_rate,
                       double regularization_constant) {
    // Reset variable derivatives.
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      if (layers[k] == output_layer_) continue;
      layers[k]->ResetOutputDerivatives();
    }

    // Backprop.
    output_layer_->set_output_labels(output_labels);
    for (int k = layers.size() - 1; k >= 0; --k) {
      layers[k]->RunBackward();
    }
  }

  void ResetParameterGradients() {
    // Reset parameter gradients.
    parameters_->ResetGradients();
#if 0
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->ResetGradients();
    }
#endif
  }

  void UpdateParameters(int batch_size,
                        double learning_rate,
                        double regularization_constant) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = layers.size() - 1; k >= 0; --k) {
      // Update parameters.
      if (use_ADAM_) {
        layers[k]->UpdateParametersADAM(batch_size,
                                        learning_rate,
                                        regularization_constant);
      } else {
        layers[k]->UpdateParameters(batch_size,
                                    learning_rate,
                                    regularization_constant);
      }
    }
  }

 protected:
  Dictionary *source_dictionary_;
  Dictionary *target_dictionary_;
  LabelAlphabet *label_alphabet_;
  Updater<Real> *updater_;
  Parameters<Real> *parameters_;
  LookupLayer<Real> *source_lookup_layer_;
  LinearLayer<Real> *source_linear_layer_;
  LookupLayer<Real> *target_lookup_layer_;
  LinearLayer<Real> *target_linear_layer_;
  RNNLayer<Real> *source_rnn_layer_;
  RNNLayer<Real> *target_rnn_layer_;
  AttentionLayer<Real> *attention_layer_;
  AlignmentLayer<Real> *alignment_layer_;
  ConcatenatorLayer<Real> *concatenator_layer_;
  FeedforwardLayer<Real> *feedforward_layer_;
  SoftmaxLayer<Real> *output_layer_;
  int embedding_size_;
  int input_size_;
  int hidden_size_;
  int output_size_;
  bool cost_augmented_;
  double cost_false_positives_;
  double cost_false_negatives_;
  bool use_attention_;
  int attention_type_;
  bool use_lstms_;
  bool use_bidirectional_rnns_;
  bool apply_dropout_;
  double dropout_probability_;
  bool use_ADAM_;
  bool write_attention_probabilities_;
  bool test_;
  std::string model_prefix_;
  std::ofstream os_attention_;
};

#endif /* ATTENTIVE_RNN_H_ */

