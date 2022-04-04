import pandas as pd
import preprocessor as p
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# nltk also has several different tokenizers
# import nltk

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER)


class TextProcessingPipeline:
    def __init__(
        self, filepath, columns=None, header=None, target="target", stopwords=None
    ):
        self.filepath = filepath
        self.columns = columns
        self.header = header
        self.data = None
        self.target = target
        self.stopwords = stopwords
        self.tokenizer = Tokenizer()
        self.X_tokenized = None
        self.X_pad = None

    def load(self):
        try:
            self.data = pd.read_csv(self.filepath, header=self.header)
        except Exception as e:
            raise Exception(f"Error in load() \n{str(e)}")
        if self.columns:
            self.data.columns = self.columns
        print(f"Data loaded with length: {len(self.data)}")
        return None

    def clean(self, stopwords=None):
        if self.data is None:
            self.load()
        if stopwords is not None:
            self.stopwords = stopwords
        print("Starting Data Cleaning...")
        self.data["clean_text"] = self.data.text.apply(
            lambda text: self.clean_text(text)
        )
        print(f"Data cleaning complete!")
        return None

    def clean_text(self, text):
        """cleans text of urls, emojis, converts to lowercase, removes stopwords if provided

        Args:
            text (str): tweet text
            stopwords (list, optional): list of stopwords. Defaults to nltk.english.stopwords.

        Returns:
            list: list of words in tweet after preprocessing
        """
        text = p.clean(text).lower()
        text = re.sub(r"\.{3,}", " ", text)
        text = re.sub(r"&[A-Za-z]*;", "", text)
        text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
        if self.stopwords:
            return " ".join(
                [word for word in text.split(" ") if word not in self.stopwords]
            )
        return text

    def tokenize_pad(self, pad=True, padding_length=20):
        # tokenizer can be swapped out
        if "clean_text" not in self.data.columns:
            self.clean()
        self.tokenizer.fit_on_texts(self.data.clean_text)
        self.X_tokenized = self.tokenizer.texts_to_sequences(self.data.clean_text)
        if pad:
            # pad sequences to same length. This is required for nearly all models
            self.X_pad = pad_sequences(
                self.X_tokenized, padding_length, padding="pre", truncating="pre"
            )
            print(f"Data Tokenized and Padded!")
            return self.X_pad
        print("Data Tokenized, but not padded")
        return self.X_tokenized

    def get_train_test(self):
        # get X whether that is padded, tokenized or if not yet processed
        if self.X_pad is not None:
            X = self.X_pad
        elif self.X_tokenized is not None:
            X = self.X_tokenized
        else:
            X = self.tokenize_pad()

        return train_test_split(
            X, self.data[self.target].to_numpy(), test_size=0.3, random_state=42
        )


if __name__ == "__main__":
    # just to make sure it works, and an example of use
    data_path = "/Users/bradpayne/Desktop/CS6890/project/data/training.1600000.processed.noemoticon.csv"
    pipeline = TextProcessingPipeline(
        filepath=data_path, columns=["target", "id", "date", "flag", "user", "text"]
    )
    # nltk.download("stopwords")
    # stopwords = nltk.corpus.stopwords.words("english")
    pipeline.load()
    X1, X2, y1, y2 = pipeline.get_train_test()
    assert len(X1) == len(y1), "ERROR: train X, y length don't match"
