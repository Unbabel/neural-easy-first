#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <assert.h>
#include "attentive_quality_estimator.h"

typedef AttentiveQualityEstimator<double> DoubleAttentiveQualityEstimator;

const int kMaxAffixSize = 4;

void LoadWordVectors(const std::string &word_vector_file,
                     std::unordered_map<std::string,
                                        std::vector<double> > *word_vectors) {
  // Read word vectors.
  std::cout << "Loading word vectors..." << std::endl;
  std::ifstream is;
  is.open(word_vector_file.c_str(), std::ifstream::in);
  assert(is.good());
  int num_dimensions = -1;
  std::string line;
  if (is.is_open()) {
    getline(is, line); // Skip first line.
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, " ", &fields);
      if (num_dimensions < 0) {
        num_dimensions = fields.size()-1;
        std::cout << "Number of dimensions: " << num_dimensions << std::endl;
      } else {
        assert(num_dimensions == fields.size()-1);
      }
      std::string word = fields[0];
      std::vector<double> word_vector(num_dimensions, 0.0);
      for (int i = 0; i < num_dimensions; ++i) {
        word_vector[i] = atof(fields[1+i].c_str());
      }
      assert(word_vectors->find(word) == word_vectors->end());
      (*word_vectors)[word] = word_vector;
    }
  }
  is.close();
  std::cout << "Loaded " << word_vectors->size() << " word vectors."
            << std::endl;
}

void ReadDataset(const std::string &dataset_file,
                 bool locked_alphabets,
                 int cutoff,
                 Dictionary *source_dictionary,
                 Dictionary *target_dictionary,
                 LabelAlphabet *label_alphabet,
                 std::vector<SentencePair> *sentence_pairs,
                 std::vector<std::vector<int> > *output_labels) {
  bool finish_with_separator = false;
  // Read data.
  std::ifstream is;
  is.open(dataset_file.c_str(), std::ifstream::in);
  assert(is.good());
  std::vector<std::vector<std::string> > sentence_fields;
  std::string line;
  std::vector<Sentence*> all_source;
  std::vector<Sentence*> all_target;
  std::vector<std::vector<std::pair<int, int> > > all_alignments;
  std::vector<std::vector<std::string> > all_labels;
  if (is.is_open()) {
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, "\t", &fields);
      //if (fields.size() != 3) std::cout << line << std::endl;
      assert(fields.size() == 3 || fields.size() == 4);
      std::string labels = fields[0];
      std::string source = fields[1];
      std::string target = fields[2];
      std::string align = (fields.size() == 4)? fields[3] : "";
      std::vector<std::string> word_labels;
      std::vector<std::string> source_words;
      std::vector<std::string> target_words;
      std::vector<std::string> aligned_pairs;
      StringSplit(labels, " ", &word_labels);
      StringSplit(source, " ", &source_words);
      StringSplit(target, " ", &target_words);
      StringSplit(align, " ", &aligned_pairs);
      if (finish_with_separator) {
        source_words.push_back("__START__");
        target_words.push_back("__START__");
      }
      std::vector<std::pair<int, int> > alignments;
      for (auto aligned_pair: aligned_pairs) {
        std::vector<std::string> pairs;
        StringSplit(aligned_pair, "-", &pairs);
        assert(pairs.size() == 2);
        int s = atoi(pairs[0].c_str());
        int t = atoi(pairs[1].c_str());
        alignments.push_back(std::make_pair(s, t));
      }
      Sentence *sentence = new Sentence;
      sentence->Initialize(source_words, ""); // Dummy label.
      all_source.push_back(sentence);
      sentence = new Sentence;
      sentence->Initialize(target_words, ""); // Dummy label.
      all_target.push_back(sentence);
      all_labels.push_back(word_labels);
      all_alignments.push_back(alignments);
      for (auto label: word_labels) {
        label_alphabet->InsertLabel(label);
      }
    }
  }
  is.close();

  // Load dictionary if necessary.
  if (!locked_alphabets) {
    source_dictionary->AddWordsFromDataset(all_source, cutoff, 0);
    target_dictionary->AddWordsFromDataset(all_target, cutoff, 0);
  }

  // Create numeric dataset.
  sentence_pairs->clear();
  output_labels->clear();
  for (int i = 0; i < all_source.size(); ++i) {
    Sentence *source_sentence = all_source[i];
    std::vector<Input> source_word_sequence(source_sentence->Size());
    for (int j = 0; j < source_sentence->Size(); ++j) {
      const std::string &word = source_sentence->GetWord(j);
      source_word_sequence[j].Initialize(word, kMaxAffixSize,
                                         *source_dictionary);
    }

    Sentence *target_sentence = all_target[i];
    std::vector<Input> target_word_sequence(target_sentence->Size());
    //const std::string &label = sentence->GetLabel();
    //int lid = dictionary->GetLabelId(label);
    for (int j = 0; j < target_sentence->Size(); ++j) {
      const std::string &word = target_sentence->GetWord(j);
      target_word_sequence[j].Initialize(word, kMaxAffixSize,
                                         *target_dictionary);
    }

    SentencePair sentence_pair;
    sentence_pair.set_source_sentence(source_word_sequence);
    sentence_pair.set_target_sentence(target_word_sequence);
    sentence_pair.set_alignments(all_alignments[i]);
    sentence_pairs->push_back(sentence_pair);

    std::vector<std::string> word_labels = all_labels[i];
    std::vector<int> word_label_ids(word_labels.size());
    for (int j = 0; j < word_labels.size(); ++j) {
      const std::string &label = word_labels[j];
      int lid = label_alphabet->GetLabelId(label);
      word_label_ids[j] = lid;
    }
    output_labels->push_back(word_label_ids);
  }
}

int main(int argc, char** argv) {
  std::string mode = argv[1]; // "train" or "test".
  std::string train_file = argv[2];
  std::string dev_file = argv[3];
  std::string test_file = argv[4];
  std::string source_word_vector_file = argv[5];
  std::string target_word_vector_file = argv[6];
  double cost_false_positives = atof(argv[7]);
  double cost_false_negatives = atof(argv[8]);
  bool use_attention = static_cast<bool>(atoi(argv[9]));
  int attention_type = atoi(argv[10]);
  int num_hidden_units = atoi(argv[11]);
  int warm_start_on_epoch = atoi(argv[12]);
  int num_epochs = atoi(argv[13]);
  int batch_size = atoi(argv[14]);
  double learning_rate = atof(argv[15]);
  double regularization_constant = atof(argv[16]);
  std::string model_prefix = argv[17];

  //int embedding_dimension = 300; //64;
  int word_cutoff = 1;

  if (use_attention) {
    if (attention_type == AttentionTypes::SPARSEMAX) {
      std::cout << "Using sparse-max attention." << std::endl;
    } else if (attention_type == AttentionTypes::SOFTMAX) {
      std::cout << "Using soft-max attention." << std::endl;
    } else { // LOGISTIC.
      std::cout << "Using logistic attention." << std::endl;
    }
  } else {
    std::cout << "Not using attention." << std::endl;
  }

  Dictionary source_dictionary, target_dictionary;
  LabelAlphabet label_alphabet;
  source_dictionary.Clear();
  source_dictionary.set_max_affix_size(kMaxAffixSize);
  target_dictionary.Clear();
  target_dictionary.set_max_affix_size(kMaxAffixSize);

  // Load the source embeddings.
  std::unordered_map<std::string, std::vector<double> > source_word_vectors;
  LoadWordVectors(source_word_vector_file, &source_word_vectors);
  int num_fixed_source_embeddings = source_word_vectors.size();
  int embedding_dimension = source_word_vectors.begin()->second.size();
  std::cout << "Original embedding dimension: " << embedding_dimension
            << std::endl;
  DoubleMatrix fixed_source_embeddings =
    DoubleMatrix::Zero(embedding_dimension, num_fixed_source_embeddings);
  for (auto it = source_word_vectors.begin(); it != source_word_vectors.end();
       ++it) {
    int wid = source_dictionary.AddWord(it->first);
    const std::vector<double> &word_vector = it->second;
    for (int k = 0; k < word_vector.size(); ++k) {
      fixed_source_embeddings(k, wid) = static_cast<float>(word_vector[k]);
    }
  }

  // Load the target embeddings.
  std::unordered_map<std::string, std::vector<double> > target_word_vectors;
  LoadWordVectors(target_word_vector_file, &target_word_vectors);
  int num_fixed_target_embeddings = target_word_vectors.size();
  assert(embedding_dimension == target_word_vectors.begin()->second.size());
  DoubleMatrix fixed_target_embeddings =
    DoubleMatrix::Zero(embedding_dimension, num_fixed_target_embeddings);
  for (auto it = target_word_vectors.begin(); it != target_word_vectors.end();
       ++it) {
    int wid = target_dictionary.AddWord(it->first);
    const std::vector<double> &word_vector = it->second;
    for (int k = 0; k < word_vector.size(); ++k) {
      fixed_target_embeddings(k, wid) = static_cast<float>(word_vector[k]);
    }
  }

  std::vector<SentencePair> sentence_pairs;
  std::vector<std::vector<int> > output_labels;
  ReadDataset(train_file, false,
              word_cutoff, &source_dictionary, &target_dictionary,
              &label_alphabet, &sentence_pairs, &output_labels);

  std::vector<SentencePair> sentence_pairs_dev;
  std::vector<std::vector<int> > output_labels_dev;
  ReadDataset(dev_file, true,
              -1, &source_dictionary, &target_dictionary, &label_alphabet,
              &sentence_pairs_dev, &output_labels_dev);

  std::vector<SentencePair> sentence_pairs_test;
  std::vector<std::vector<int> > output_labels_test;
  ReadDataset(test_file, true,
              -1, &source_dictionary, &target_dictionary, &label_alphabet,
              &sentence_pairs_test, &output_labels_test);

  std::cout << "Number of sentences: " << sentence_pairs.size() << std::endl;
  std::cout << "Number of source words: " << source_dictionary.GetNumWords()
            << std::endl;
  std::cout << "Number of target words: " << target_dictionary.GetNumWords()
            << std::endl;
  std::cout << "Number of labels: " <<  label_alphabet.GetNumLabels()
            << std::endl;

  DoubleAttentiveQualityEstimator quality_estimator(&source_dictionary,
                                                    &target_dictionary,
                                                    &label_alphabet,
                                                    embedding_dimension,
                                                    num_hidden_units,
                                                    label_alphabet.
                                                      GetNumLabels(),
                                                    cost_false_positives,
                                                    cost_false_negatives,
                                                    use_attention,
                                                    attention_type);
  //rnn.SetFixedEmbeddings(fixed_embeddings, word_ids);
  quality_estimator.SetModelPrefix(model_prefix);
  quality_estimator.SetFixedEmbeddings(fixed_source_embeddings,
                                       fixed_target_embeddings);

  if (mode == "train") {
    //quality_estimator.InitializeParameters();
    quality_estimator.Train(sentence_pairs, output_labels,
                            sentence_pairs_dev, output_labels_dev,
                            sentence_pairs_test, output_labels_test,
                            warm_start_on_epoch,
                            num_epochs, batch_size, learning_rate,
                            regularization_constant);
  } else {
    // mode == "test".
    //quality_estimator.InitializeParameters();
    quality_estimator.LoadModel(model_prefix);
    quality_estimator.Test(sentence_pairs_dev, output_labels_dev,
                           sentence_pairs_test, output_labels_test);
  }
}
