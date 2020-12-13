import logging.config

from code import store_model
from code.config import logger_config

logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


class NewsV2Model:
    def __init__(self):
        """
        Initializes the model by loading the stored pickle.
        """
        model_data = store_model.load_model_data()
        self.model = model_data['model']
        self.model_status = model_data['metadata']

    def predict(self, message_df):
        """
        Performs a prediction for the given message
        :param message_df: message as a dataframe.
        :return: the string label
        """
        prediction = self.model.predict(message_df)[0]
        return prediction

    def get_model_status(self):
        """
        Returns the accuracy score
        :return: accuracy score
        """
        return {
            'accuracy': self.model_status['accuracy'],
        }