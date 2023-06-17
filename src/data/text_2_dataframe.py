"""
This module provides the Text2DF class, which is used to convert text data into a pandas DataFrame. 
The Text2DF class performs various text processing operations such as 
cleaning, filtering, and vectorizing texts. 
It supports options to use a predefined wordlist and stopwords for filtering,
as well as n-grams generation using the CountVectorizer from scikit-learn.
Usage:
    t2df = Text2DF(keep_words=["good", "bad"], vectorizer=True, ngrams_range=(1, 2), 
                   use_wordlist=True, use_stopwords=True)
    cleaned_texts = t2df.clean_texts(texts)
    filtered_texts = t2df.filter_texts(texts, wordlist)
    dataframe = t2df.make_dataframe(texts, use_filtered_wordlist=True)

"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Text2DF():
    """
    Text2DF is a class for converting text data into a pandas DataFrame.

    Args:
        keep_words (list): A list of words to include in the DataFrame.
        vectorizer (bool): Indicates whether to use CountVectorizer or manual word counting.

    Attributes:
        CV (CountVectorizer): CountVectorizer object for text vectorization.
        wordlist (list): List of English 10000 most common words.
        stopwords (list): List of English most used stopwords.
        keep_words (list): List of words to include in the DataFrame.
        vectorizer (bool): Indicates whether to use CountVectorizer or manual word counting.

    """
    count_vectorizer: CountVectorizer
    wordlist: list
    stopwords: list
    keep_words: list
    vectorizer: bool

    # list of english 10000 most common words
    _wordlist_10000_file_path = os.path.abspath('../data/external/list_10000_words.txt')

    # list of english most used stopwords
    _stopwords_file_path = os.path.abspath('../data/external/stopwords.txt')

    def __init__(self, keep_words=None, vectorizer=False, **kwargs):
        """
        Initialize the Text2DF object.

        Args:
            keep_words (list): A list of words to include in the DataFrame.
            vectorizer (bool): Indicates whether to use CountVectorizer or manual word counting.
            kwargs: Additional keyword arguments for configuring CountVectorizer.

        Raises:
            AssertionError: If vectorizer is set to True but required 
            keyword arguments are not provided.

        """
        self.wordlist = self._get_wordlist()
        self.stopwords = self._get_stopwords()
        self.keep_words = keep_words
        self.vectorizer= vectorizer
        if vectorizer:
            assert {'ngrams_range', 'use_stopwords', 'use_wordlist'}.issubset(
                kwargs.keys()), "specify ngrams_range, use_wordlist, and use_stopwords args"
            assert isinstance(kwargs['ngrams_range'], tuple), "ngrams_range: tuple (int, int)"
            if kwargs['use_stopwords'] and kwargs['use_wordlist']:
                self.count_vectorizer = CountVectorizer(ngram_range=kwargs['ngrams_range'], vocabulary=self.wordlist, stop_words=self.stopwords)
            elif kwargs['use_stopwords']:
                self.count_vectorizer = CountVectorizer(ngram_range=kwargs['ngrams_range'], stop_words=self.stopwords)
            elif kwargs['use_wordlist']:
                self.count_vectorizer = CountVectorizer(ngram_range=kwargs['ngrams_range'], vocabulary=self.wordlist)
            else:
                self.count_vectorizer = CountVectorizer(ngram_range=kwargs['ngrams_range'])

    def _get_wordlist(self):
        """
        Retrieve the list of English 10000 most common words.

        Returns:
            list: The list of English 10000 most common words.

        """
        # opening the file
        with open(self._wordlist_10000_file_path, 'r', encoding='utf-8') as file:
            wordlist = file.readlines()
        file.close()

        # stripping the new line character
        return [w.strip('\n') for w in wordlist]

    def _get_stopwords(self):
        """
        Retrieve the list of English most used stopwords.

        Returns:
            list: The list of English most used stopwords.

        """
        # opening the file
        with open(self._stopwords_file_path, 'r', encoding='utf-8') as file:
            stopwords = file.readlines()
        file.close()

        # stripping the new line character
        return [w.strip('\n') for w in stopwords]

    def clean_texts(self, texts):
        """
        Clean the given texts by removing usernames, hashtags, links, special characters,
        converting contractions, and removing multiple newlines and spaces.

        Args:
            texts (str or list): The input texts to clean.

        Returns:
            pd.Series: A pandas Series object containing the cleaned texts.

        """
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)

        return texts.apply(self._clean_text)

    def filter_texts(self, texts, wordlist):
        """
        Filter the given texts by keeping only the words present in the provided wordlist.

        Args:
            texts (str or list): The input texts to filter.
            wordlist (list): The list of words to filter by.

        Returns:
            pd.Series: A pandas Series object containing the filtered texts.

        """
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)

        return texts.apply(self._filter_text, args=(wordlist,))

    def _clean_text(self, text):
        """
        Clean the given text by removing usernames, hashtags, links, special characters,
        converting contractions, and removing multiple newlines and spaces.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.

        """
        intermed_text = re.sub(r'(@[^ ]*)+', '', text.lower())  # delete usernames
        intermed_text = re.sub(r'(#[^ ]*)+', '', intermed_text)  # delete #tags
        intermed_text = re.sub(r'(http[^ ]*)+', '', intermed_text)  # get rid of links
        intermed_text = re.sub(r'[\d$.!?<>/\\,\.\(\)\[\]\{\}\-\_`\'":%]+', ' ', intermed_text)  # remove special chars
        intermed_text = re.sub(r"n't", ' not', intermed_text)  # convert n't to not
        return re.sub(r'[ \n]+', ' ', intermed_text).strip()  # get rid of multiple newlines and spaces

    def _filter_text(self, text, wordlist):
        """
        Filter the given text by keeping only the words present in the provided wordlist.

        Args:
            text (str): The input text to filter.
            wordlist (list): The list of words to filter by.

        Returns:
            str: The filtered text.

        """
        text_split = text.split()
        return ' '.join([w for w in text_split if w in wordlist])

    def make_dataframe(self, texts, use_filtered_wordlist=False):
        """
        Create a pandas DataFrame from the given texts.

        Args:
            texts (str or list): The input texts to convert into a DataFrame.
            use_filtered_wordlist (bool): Indicates whether to use the filtered wordlist or the original wordlist.

        Returns:
            pd.DataFrame: The DataFrame representing the texts.

        Raises:
            IndexError: If a word in the texts is not present in the columns.
                You need to filter and clean your texts before making the dataframe.

        """
        if self.vectorizer:
            sparse_matrix = self.count_vectorizer.transform(texts)
            return pd.DataFrame(data=sparse_matrix.toarray(), columns=self.count_vectorizer.get_feature_names_out())

        if use_filtered_wordlist:
            data_frame = pd.DataFrame(data=np.zeros((len(texts), len(self.get_filtered_wordlist())), dtype='uint8'),
                              columns=self.get_filtered_wordlist())
        else:
            data_frame = pd.DataFrame(data=np.zeros((len(texts), len(self.wordlist)), dtype='uint8'), columns=self.wordlist)

        for i, text in enumerate(texts):
            for word_in_tweet in text.split():
                try:
                    data_frame.loc[i, word_in_tweet] += 1
                except Exception as exc:
                    raise IndexError(f"There is no word '{word_in_tweet}' in the columns.\n"
                                     "You have to filter and clean your texts before making the dataframe") from exc
        return data_frame

    def set_wordlist(self, wordlist):
        """
        Set the wordlist to be used for creating the DataFrame.

        Args:
            wordlist (list): The wordlist to set.

        """
        self.wordlist = wordlist

    def set_stopwords(self, stopwords):
        """
        Set the stopwords to be used for cleaning and filtering the texts.

        Args:
            stopwords (list): The stopwords to set.

        """
        self.stopwords = stopwords

    def get_stopwords(self):
        """
        Get the stopwords being used for cleaning and filtering the texts.

        Returns:
            list: The stopwords.

        """
        return self.stopwords

    def get_wordlist(self):
        """
        Get the wordlist being used for creating the DataFrame.

        Returns:
            list: The wordlist.

        """
        return self.wordlist

    def get_filtered_wordlist(self):
        """
        Get the filtered wordlist based on stopwords and additional keep_words.

        Returns:
            list: The filtered wordlist.

        """
        return list(set(w for w in self.wordlist if w not in self.stopwords).union(set(self.keep_words)))
