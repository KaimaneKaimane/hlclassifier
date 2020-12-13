import logging

import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)


def news_v2(columns: list) -> pd.DataFrame:
    """
    Loads the entire news_v2 dataset. Drops empty headlines
    :param columns: preselects some columns during loading
    :return: the news_v2 dataset as a dataframe
    """

    dataset_path = Path.cwd() / Path('dataset/News_Category_Dataset_v2.json')
    logger.info(dataset_path)
    df = pd.read_json(dataset_path, lines=True)

    reduced_df = df.drop(df[df['headline'].str.len() == 0].index, axis=0)
    reduced_df = reduced_df[columns]

    return reduced_df