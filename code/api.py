import json
import logging.config

import fastjsonschema
import pandas as pd
from flask import Flask, request, Response

from code import preprocessing
from code.config import config, json_request_schema, logger_config
from code.predict import NewsV2Model

app = Flask(__name__)

if __name__ != '__main__':
    logging.config.dictConfig(logger_config)
    logger = logging.getLogger(__name__)

    model = NewsV2Model()

    # json validator
    validate_json_request = fastjsonschema.compile(json_request_schema)

    logger.info('Waiting for requests')


@app.route('/predict', methods=['POST'])
def predict() -> Response:
    """
    Predicts the category of the input data.
    :return: JSON Response containing the prediction result
    """
    logger.debug('Request received')

    message = request.get_json()
    validate_json_request(message)

    message_df = pd.DataFrame(
        {key: [value] for key, value in message.items()}
    )
    preprocessed_message = preprocessing.run(message_df, training=False)
    prediction = {config['dataset']['label_column']: model.predict(preprocessed_message['headline'].tolist())}
    logger.info(prediction)
    return Response(json.dumps(prediction), 200)


@app.route('/status')
def status() -> Response:
    """
    Returns the status of the current model.
    :return: JSON Response containing the model status
    """

    model_status = model.get_model_status()

    if model_status is not None:
        return_code = 200
    else:
        model_status = {}
        return_code = 503

    return Response(
        json.dumps(model_status),
        return_code,
        mimetype='application/json'
    )


@app.errorhandler(500)
def internal_server_error(error: int) -> Response:
    """
    Handle erros with code 500.
    :param error: internal server error
    :return: Response containing the internal server error exception
    """
    try:
        logger.exception('Server error: %s', error)
    except Exception:
        pass
    return Response('Internal server error', 500)


@app.errorhandler(fastjsonschema.JsonSchemaException)
def json_schema_exception(error: Exception) -> Response:
    """
    Called when a JsonSchema-Validation fails.
    :param error: the json schema exception
    :return: Response containing the exception
    """
    return Response('Bad request: {} '.format(str(error)), 400)


@app.errorhandler(Exception)
def unhandled_exception(error: Exception) -> Response:
    """
    Called when an unhandled exception occurs.
    :param error: the unhandled exception
    :return: Response containing the unhandled exception
    """
    try:
        logger.exception('Unhandled Exception: %s', error)
    except Exception:
        pass
    return Response('Internal server error', 500)