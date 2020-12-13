import pandas as pd
import re

from code.config import config

def _filter_columns(dataset: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Filters out certain columns which are not required for the prediction.
    :param dataset: the news_v2 dataset
    :return: the reduced news_v2 dataset
    """
    filter_columns = config['dataset']['filter_columns']
    if not training and config['dataset']['label_column'] in filter_columns:
        filter_columns.remove(config['dataset']['label_column'])

    return dataset[filter_columns]


def _clean_headline(original_headline):
    """
    Remove special characters, multiple sequential spaces from string and transform to lowercase
    :param original_headline: the original string
    :return: the transformed string
    """
    cleaned_headline = re.sub(r'\W', ' ', original_headline)
    cleaned_headline = re.sub(r'\s+', ' ', cleaned_headline)

    cleaned_headline = cleaned_headline.lower()
    return cleaned_headline


def _preprocess_headline(dataset: pd.DataFrame):
    """
    Cleans the headline string.
    :param dataset: the news_v2 dataset containing the headline column
    :return: the news_v2 dataset with the cleaned headline
    """
    dataset['headline'] = dataset['headline'].apply(_clean_headline)
    return dataset


def run(dataset: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """
    performs all the preprocessing steps for the news_v2 data.
    :param dataset: the news_v2 dataset
    :param training: is the preprocessing performed for training or prediction
    :return: the preprocessed dataset
    """
    dataset = _filter_columns(dataset, training)
    dataset = _preprocess_headline(dataset)

    return dataset