import logging
import pickle as pkl
from pathlib import Path

model_filepath = Path(__file__).parents[1].resolve() / Path('model/news_v2_model.pkl')

logger = logging.getLogger(__name__)


def save_model_data(model_data):
    """
    Saves the model data as a pickle.
    :param model_data: the model data
    """
    logger.info('Saving model data...')
    with open(model_filepath, 'wb+') as file_handle:
        pkl.dump(model_data, file_handle)


def load_model_data():
    """
    Loads the model data from a pickle file.
    :returns: the model data
    """
    logger.info('Loading model data...')
    logger.info(model_filepath)
    with open(model_filepath, 'rb') as file_handle:
        return pkl.load(file_handle)