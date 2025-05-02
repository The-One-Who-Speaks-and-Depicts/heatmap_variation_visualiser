import logging

from configuration import set_configuration
from data_preprocessing.data_preprocessing_pipeline import data_preprocessing_pipeline
from visualisation.heatmap import create_heatmap

logger = logging.getLogger("heatmap_creator")

def run_pipeline(args):
    config = set_configuration(args)
    data = data_preprocessing_pipeline(config)
    logger.debug("Current results is %s syntagmae", len(data))
    if config.data_type == "text_variation":
        for idx, val in enumerate(data):
            create_heatmap(val, config.heatmap_params, idx)
    else:
        create_heatmap(data, config.heatmap_params)
    logger.debug("Execution finished")
    