import logging

from configuration import set_configuration
from data_preprocessing.data_preprocessing_pipeline import data_preprocessing_pipeline
from visualisation.heatmap import create_heatmap, create_heatmap_with_labels, create_heatmap_from_matrix

logger = logging.getLogger("heatmap_creator")

def run_pipeline(args):
    config = set_configuration(args)
    data = data_preprocessing_pipeline(config)
    logger.debug("Current results is %s texts", len(data))
    if config.data_type == "lemma" or "text_variation" in config.data_type:
        for idx, val in enumerate(data[0]):
            create_heatmap_with_labels(val, data[1][idx], dataset_part_idx=idx)
    elif config.data_type == "pos":
        params = config.heatmap_params
        for idx, val in enumerate(data):
            create_heatmap_from_matrix(
                    val, 
                    params.output_dir, params.is_continuous,
                    idx, idx
                    )
   
    else:
        create_heatmap(data, config.heatmap_params)
    logger.debug("Execution finished")
    