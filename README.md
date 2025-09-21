# turkish-text-classification
This repository contains sample code for training classifiers on the **OffensEval-TR 2020** dataset, a specialized version of the popular OffensEval 2020 dataset for Turkish. The goal of this project is to demonstrate different methods for offensive language classification using various techniques and models.

### Project Overview

This project includes three Jupyter Notebooks, each using different techniques to train classifiers. The notebooks are designed to act as sample code to demonstrate how to approach offensive language classification with various embeddings and models, rather than focusing on achieving high accuracy.

#### Notebooks Overview:

1. **`distil_bert_classification.ipynb`**

   * Finetunes a Turkish language variant of **DistilBERT** model 
    to classify offensive vs. non-offensive tweets.

2. **`Gemma_embedding_classification.ipynb`**

   * Two parts:

     1. **Zero-shot classifier** using the **EmbeddingGemma** model for offensive language detection.
     2. Trains an **SVM classifier** using **EmbeddingGemma**'s output.

     * **Note**: The performance of the **EmbeddingGemma** model in this case may not be optimal, but this notebook demonstrates how to use it for the task.

3. **`Fasttext_embedding_classification.ipynb`**

   * Trains a classifier using **FastText** word embeddings, utilizing pre-trained **Turkish word vectors** to classify tweets into offensive or not.

### Notes:

* The project aims to provide basic sample code for training models on the dataset, without optimizing for performance.
* The dataset is highly imbalanced, which may lead to skewed performance metrics. (You should check f1 and weighted accuracy!)
* The focus is on demonstrating the use of various classification techniques rather than achieving the highest accuracy.


### Dataset: [stefan-it/offenseval2020\_tr](https://huggingface.co/datasets/stefan-it/offenseval2020_tr)

The dataset is highly imbalanced and consists of annotated tweets for offensive language identification. The tweets are labeled as:

* NOT - Not offensive
* OFF - Offensive

The dataset is split as follows:

* **Training dataset**: 30,000 tweets
* **Development dataset**: 1,756 tweets
* **Test dataset**: 3,528 tweets

### Results

Since the dataset is imbalanced, the accuracy scores may be lower than expected, especially for **EmbeddingGemma**. The purpose of this project is to show how different techniques can be applied to offensive language detection in Turkish, and not necessarily to maximize classification performance.

### Acknowledgements

* **OffensEval 2020**: The original dataset for this task.
* **DistilBERT**: A smaller, faster version of BERT, optimized for various NLP tasks.
* **FastText**: Word embeddings for text classification.
* **EmbeddingGemma**: Zero-shot model used for generating embeddings.
