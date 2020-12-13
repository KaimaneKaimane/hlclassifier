import logging.config
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from code.spacy_model import SpacyModel
from code.config import config, logger_config
from code import load_data, preprocessing, store_model


logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


def prepare_data():
    """
    Loads & preprocessed the news_v2 data.
    :return: the preprocessed news_v2 data
    """
    logger.info('Preprocessing data...')
    dataset = load_data.news_v2(config['load']['extract_columns'])
    dataset = preprocessing.run(dataset, training=True)

    return dataset


def perform_train_test_split(dataset):
    """
    Creates a train test split.
    :param dataset: the news_v2 dataset
    :return: the train and test set
    """
    logger.info('Create train val test split...')
    features = dataset['headline']

    logger.info(config['dataset']['label_column'])
    labels = dataset[config['dataset']['label_column']]

    return train_test_split(
        np.array(features),
        np.array(labels),
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )


def train_model(x_train_set, y_train_set):
    """
    Trains a model.
    :param x_train_set: the news_v2 features
    :param y_train_set: the news_v2 labels
    :return: the trained model
    """
    logger.info('Train  model...')
    spacy_model_trainer = SpacyModel()
    return spacy_model_trainer.train_model(x_train_set, y_train_set)


def evaluate_model(x_test_set, y_test_set, model):
    """
    Evaluates the given model.
    :param x_test_set: the news_v2 features
    :param y_test_set: the news_v2 labels
    :param model: the model to evaluate
    :return: dict containing the metrics
    """
    logger.info('Evaluate model...')
    metadata = {}

    predicted_labels = model.predict(x_test_set)
    true_labels = y_test_set

    metadata['accuracy'] = accuracy_score(true_labels, predicted_labels)

    return metadata


def run():
    """
    Trains and saves the model.
    """
    news_v2_dataset = prepare_data()
    x_train_set, x_test_set, y_train_set, y_test_set = perform_train_test_split(news_v2_dataset)
    news_v2_category_model = train_model(x_train_set, y_train_set)
    news_v2_metadata = evaluate_model(x_test_set, y_test_set, news_v2_category_model)

    news_v2_model_data = {
        'model': news_v2_category_model,
        'metadata': news_v2_metadata
    }

    store_model.save_model_data(news_v2_model_data)


if __name__ == '__main__':
    run()