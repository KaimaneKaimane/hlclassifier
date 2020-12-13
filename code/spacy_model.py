import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def clean_text(text):
    """
    Removes whitespaces from start and end and transforms to lowercase
    :param text: initial text
    :return: cleaned text
    """
    return text.strip().lower()


class Predictors(TransformerMixin):
    def transform(self, x, **transform_params):
        """
        Performs the text transformation
        :param x: the texts to transform
        :param transform_params: additional transformation parameters, if required
        :return: the transformed texts
        """
        return [clean_text(text) for text in x]

    def fit(self, x, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


class SpacyModel():
    def __init__(self):
        """
        Initializes the spacy model for training
        """
        self.punctuations = string.punctuation

        # Load spacy stopwords
        self.nlp = spacy.load('en')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

        # Load English tokenizer, tagger, parser, NER and word vectors
        self.parser = English()

        self.vectorizer = TfidfVectorizer(
            tokenizer=self._spacy_tokenizer,
            min_df=1,
            ngram_range=(1, 5),
        )

        # classifier = LogisticRegression()
        self.classifier = LinearSVC()

    def _spacy_tokenizer(self, sentence):
        """
        Tokenize and lemmatize words for spacy
        :param sentence: the input sentence
        :return: the created tokens
        """
        mytokens = self.parser(sentence)
        mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [word for word in mytokens if word not in self.stop_words and word not in self.punctuations]

        return mytokens

    def train_model(self, x_train, y_train):
        """
        Combine the cleaner, vectorizer and classifier to train the model
        :param x_train: training features
        :param y_train: training labels
        :return: the trained spacy model
        """

        pipe = Pipeline([
            ('cleaner', Predictors()),
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])

        pipe.fit(x_train, y_train)

        return pipe
