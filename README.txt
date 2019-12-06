Homework solutions of CSCI 544: Natural Language Processing at the University of Southern California

1) PA1: Naive Bayes text classification
- Assignment:
In this assignment you will write a naive Bayes classifier to identify hotel reviews as either truthful or deceptive, and either positive or negative. You will be using the word tokens as features for classification

- Approach:
+ Perform tokenization: separate words, lowercase all words,
+ Calculate posterior based on prior and conditional probabilites
P(class|words) ~ P(words|class)*P(class) ~ P(word1|class)^k1 *P(word2|class)^k2...*P(classs)

2) PA2: Perceptron Model to classify reviews
- Assignment:
In this assignment you will write perceptron classifiers (vanilla and averaged) to identify hotel reviews as either truthful or deceptive, and either positive or negative. You may use the word tokens as features, or any other features you can devise from the text.

- Approach
  ~ Tokenize input document --> create a document with name, text content, label, word frequency
  ~ Initialize weights and bias for each kind of models: w={word1:weight1, word2:weight2...}, b={word1:bias1,...}
  ~ Update weightWord += label*wordFrequency,
  ~ Update biasWord += label
  ~ Specify which models to use: Vanilla or Average
  ~ Classify each document by multiplying weights with frequency: 1 POS or TRU, -1 then NEG or DEC

PA3: Hidden Markov Model POS tagger
- Assignment:
In this assignment you will write a Hidden Markov Model part-of-speech tagger for Italian, Japanese, and a surprise language. The training data are provided tokenized and tagged; the test data will be provided tokenized, and your tagger will add the tags.

- Approach:
Hidden markov model (see problem 3 hw2) from a tagged corpus
  - each state is a part-of-speech tag
  - each tag can emit observations, which are words
  - transition probabilities are the conditional probabilities of tags sequence
  - emission probabilities are the conditional probabilities of words given tags.
  - The start state is the beginning of a sentence, which is not a partof-speech tag
Note that backtrack pointer applies to all state nodes (need to find for all nodes)

PA4: Deep Learning with TensorFlow
- Assignment:
The goal of this coding assignment to get you familiar with TensorFlow and walk you through some practical Deep Learning techniques. The output layer classied each document into (positive/negative) and (truthful/deceptive).

- Approach:
+ Improve the Tokenization.
+ Convert therst layer into an Embedding Layer, which makes the model somewhat more interpretable. Many recent Machine Learning eorts strive to make models more interpretable, but sometimes at the expense of prediction accuracy.
+ Increase the generalization accuracy of the model by implementing input sparse dropout.
+ Visualize the Learned Embeddings using t-SNE.

PA5: Recursive Neural Network in TensorFlow
- Assignment:
The goal of this coding assignment to get you expertise in TensorFlow, especially in developing code from the grounds-up. This assignment will be not much hand-held: you have to do most things from scratch, including creating a tf.Session. In this task, you will be implementing a sequence-to-sequence Recursive Neural Network (RNN) model in TensorFlow to perform POS tagger

- Approach:
+ Data preprocessing
+ Code sequence model
