import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Split sentence into words.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Reduce word to root form.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, vocabulary):
    """
    Convert tokens into Bag-of-Words vector.
    """

    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    vector = np.zeros(len(vocabulary), dtype=np.float32)

    for idx, word in enumerate(vocabulary):

        if word in tokenized_sentence:
            vector[idx] = 1.0

    return vector