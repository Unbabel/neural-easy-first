#ifndef LOOKUP_LAYER_H_
#define LOOKUP_LAYER_H_

#include "layer.h"

template<typename Real> class LookupLayer : public Layer<Real> {
 public:
  LookupLayer(int num_words, int embedding_dimension) {
    this->name_ = "Lookup";
    num_words_ = num_words;
    num_words_fixed_ = 0;
    embedding_dimension_ = embedding_dimension;
  }

  virtual ~LookupLayer() {}

  int embedding_dimension() { return embedding_dimension_; }

  virtual void CreateParameters(Parameters<Real> *parameters) {
    // Only consider as parameters those embeddings that are not fixed.
    Matrix<Real> *E, *dE;
    parameters->CreateMatrixParameter("embeddings", embedding_dimension_,
                                      num_words_ - num_words_fixed_,
                                      &E, &dE);
    SetParameters(E, dE);
  }

  void SetParameters(Matrix<Real> *E,
                     Matrix<Real> *dE) {
    E_ = E;
    dE_ = dE;
  }

  void ResetParameters() {
    // Remove this.
#if 0
    // Only consider as parameters those embeddings that are not fixed.
    E_ = Matrix<Real>::Zero(embedding_dimension_,
                            num_words_ - num_words_fixed_);
#endif
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(E_);
    weight_names->push_back("embeddings");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(dE_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
    return 0.05;
  }

  void SetFixedEmbeddings(const Matrix<Real> *fixed_embeddings) {
    // Assumes the word ids of the fixed embeddings are contiguous,
    // starting in zero up to num_words_fixed_.
    E_fixed_ = fixed_embeddings;
    num_words_fixed_ = E_fixed_->cols();
  }

  void ResetGradients() {
    // Remove this.
#if 0
    dE_.setZero(embedding_dimension_, num_words_ - num_words_fixed_);
#endif
  }

  void RunForward() {
    this->GetMutableOutput()->setZero(embedding_dimension_,
                                      input_sequence_.size());
    assert(this->GetMutableOutput()->rows() == embedding_dimension_ &&
           this->GetMutableOutput()->cols() == input_sequence_.size());
    for (int t = 0; t < input_sequence_.size(); ++t) {
      int wid = input_sequence_[t];
      assert(wid >= 0 && wid < num_words_);
      if (wid >= num_words_fixed_) {
        // Dynamic embedding.
        this->GetMutableOutput()->col(t) = E_->col(wid - num_words_fixed_);
      } else {
        this->GetMutableOutput()->col(t) = E_fixed_->col(wid);
      }
    }
  }

  void RunBackward() {
    // Don't have input derivatives (they're dense and expensive and not
    // necessary.)
    for (int t = 0; t < input_sequence_.size(); ++t) {
      int wid = input_sequence_[t];
      assert(wid >= 0 && wid < num_words_);
      // Don't need to do anything for the fixed embeddings.
      if (wid >= num_words_fixed_) {
        // Dynamic embedding.
        dE_->col(wid - num_words_fixed_) += this->GetMutableOutput()->col(t);
      }
    }
  }

  void set_input_sequence(const std::vector<int> &input_sequence) {
    input_sequence_ = input_sequence;
  }

 public:
  int num_words_fixed_;
  int num_words_;
  int embedding_dimension_;
  const Matrix<Real> *E_fixed_;
  Matrix<Real> *E_;
  Matrix<Real> *dE_;
  std::vector<int> input_sequence_; // Input.
};

#endif /* LOOKUP_LAYER_H_ */
