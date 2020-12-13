from code import store_model
from code.config import config


def run():
    """
    Prints the metrics for the news_v2 model
    """
    print(config)

    model_data = store_model.load_model_data()

    print('Accuracy Score:', model_data['metadata']['accuracy'])


if __name__ == '__main__':
    run()