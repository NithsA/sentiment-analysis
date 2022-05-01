## Tweet Sentiment Analysis project

### Authors
#### Bradley Payne
#### Nithya Alavala
---
## Project Summary 
<p>
Sentiment analysis is an important task for many domains including for product feedback and how the
general public feels about a given company, product or topic. Many people express their ideas,
concerns, movie analysis, and other opinions on twitter through tweets. The goal of this project is to take texts
from twitter data and classify it as negative, neutral, or positive.

One area that we would like to explore is the use of emojis in sentiment analysis.</p>

---
## Datasets 

- [1.6 Million Tweet dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
    - Contains 800,000 positive and 800,000 negative tweets
    - labeled with 0 for negative, 4 for positive tweets
    - Dataset does NOT contain emojis

- [GloVe vectors for word representation](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation)
    - Pretrained word embedding 
---

## Models
We used the following methods

* Random Forest
* Decision Tree
* Naive Bayes
* SVM - Classifier
* LSTM
* BERT 

## SRC files
Below is a brief summary of each of the source code files
- ### [attention.ipynb](/src/attention.ipynb)
    - Contains the LSTM with the added attention layer
- ### [bert-pretrained.ipynb](/src/bert-pretrained.ipynb)
    - Implements a version of BERT from the huggingface transformers library
    - Trains BERT for 2 epochs
    - Evaluates on the test set
- ### [data_exploration.ipynb](/src/data_exploration.ipynb)
    - Our first notebook to explore the data
    - Fits the random forest as proof of concept 
- ### [data_pipeline.py](/src/data_pipeline.py)
    - Defines a data preprocessing pipeline that:
        - Loads the data
        - Cleans
        - Tokenizes
        - Splits into testing and training sets
- ### [lstm_models.ipynb](/src/lstm_models.ipynb)
    - Fits an LSTM model on:
        - Raw data
        - Preprocessed data
- ### [models.ipynb](/src/models.ipynb)
    - Uses `data_pipeline.py` functionality to get training and testing set
    - Fits `sklearn` models:
        - Naive Bayes
        - SVM
        - Decision Tree
        - Random Forest
