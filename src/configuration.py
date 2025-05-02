import logging

from dataclasses import dataclass

from corpus_distance.data_preprocessing.topic_modelling import LDAParams
from corpus_distance.pipeline import create_and_set_storage_directory

from data_preprocessing.matrix_transformation import EqualizerParams

from visualisation.heatmap import HeatmapParams

logger = logging.getLogger("heatmap_creator")

@dataclass
class ConfigurationParameters:
    data_type: str = "ua_ner"
    source_directory: str = ""
    variation_type: str = ""
    rapidity_rate: int = 3
    clustered: bool = False
    text_separator: str = ""
    equalizer_params: EqualizerParams = EqualizerParams()
    lda_params: LDAParams = LDAParams()
    heatmap_params: HeatmapParams = HeatmapParams()




def set_configuration(args) -> ConfigurationParameters:
    args = args.__dict__
    for arg in args:
        if args[arg]:
            logger.info(f"{arg}: {args[arg]}")
        else:
            logger.info(f"{arg} not given")
    cfg = ConfigurationParameters()
    if args["data_type"]:
        cfg.data_type = args["data_type"]
    if args["source_directory"]:
        cfg.source_directory = args["source_directory"]
    if args["variation_type"]:
        cfg.variation_type = args["variation_type"]
    if args["rapidity_rate"]:
        cfg.rapidity_rate = int(args["rapidity_rate"]) if args["rapidity_rate"] else 3
    if args["clustered"]:
        cfg.clustered = True if args["clustered"] == "1" else False
    if args["text_separator"]:
        cfg.text_separator = args["text_separator"]
    if args["func_path"]:
        cfg.equalizer_params.func_path = args["func_path"]
    if args["percentile"]:
        cfg.equalizer_params.percentile = int(args["percentile"])
    if args["lda_num_topics"]:
        cfg.lda_params.num_topics = int(args["lda_num_topics"])
    if args["lda_alpha"]:
        cfg.lda_params.alpha = args["lda_alpha"]
    if args["lda_passes"]:
        cfg.lda_params.passes = int(args["lda_passes"])
    if args["lda_epochs"]:
        cfg.lda_params.epochs = int(args["lda_epochs"])
    if args["lda_required_num"]:
        cfg.lda_params.required_topics_num = int(args["lda_required_num"])
    if args["lda_required_start"]:
        cfg.lda_params.required_topics_start = int(args["lda_required_start"])
    if args["output_directory"]:
        cfg.heatmap_params.output_dir = create_and_set_storage_directory(args["output_directory"])
    if args["max_rows"]:
        cfg.heatmap_params.max_rows = int(args["max_rows"])
    if args["is_heat_continuous"]:
        cfg.heatmap_params.is_continuous = True if args["is_heat_continuous"] == "True" else False
    return cfg