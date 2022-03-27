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

- [Emoji sentiment summary](https://www.kaggle.com/datasets/thomasseleck/emoji-sentiment-data)
    - result of work done [Here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144296)
    - contains the # occurences of a series of tweets in positive and negative tweets of an un-named dataset 
    - contains a word description of different emojis

- [Emoji dataset](https://github.com/zhengrui315/Twitter-Sentiment-Analysis/tree/master/data)
    - contains tweet dataset with emojis, but they are not labeled 
---

## Models and results
Our plan is to run the 1.6M dataset through the following models:

* Random Forest
* Naive Bayes
* SVM - Classifier
* KNN
* RNN

Then, we can choose the model with the best results to focus hypertuning. 

Currently, the file `data_exploration.ipynb` cleans the tweets of punctuation, stopwords, emojis, urls, mentions, and converts it to a Bag-of-word representation. 

Then it trains a classifer using 100,00 tweets as the training and 30% of the dataset as a testing set. Doing this, the random forest took 2 hours to fit and had an accuracy of 75.1%

---

## TODO
* convert the cleaning data into a python class in a `.py` file so it can be reused
* convert the BOW representation into a tokenization method since that is typically better
* fit each of the models indicated using a consistant(seeded) train/test split 
