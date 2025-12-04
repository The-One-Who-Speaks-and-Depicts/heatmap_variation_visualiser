import logging

from configuration import ConfigurationParameters
from data_preprocessing.data_loading import load_source_by_type, LemmaForHeat
from data_preprocessing.configuration_loading import load_variation_types
from data_preprocessing.heat_measurement import transform_to_heat_sequence
from data_preprocessing.matrix_transformation import equalize_row_length_in_sentence_matrix

logger = logging.getLogger('heatmap_creator')

def data_preprocessing_pipeline(cfg: ConfigurationParameters) -> list[list] | tuple[list[list], list[list]]:    
    source = load_source_by_type(
        cfg.data_type, 
        source_dir = cfg.source_directory, 
        text_separator = cfg.text_separator, 
        lda_params = cfg.lda_params,
        rapidity_rate = cfg.rapidity_rate
        )
    variation_types = load_variation_types(cfg.variation_type)
    if cfg.data_type == "text_variation":
        heat_texts = []
        for t in source:
            heat_texts.append(
                list(
                    transform_to_heat_sequence(
                    s, cfg.data_type, variation_types, cfg.rapidity_rate, cfg.clustered) for s in t
                )
            )
        equalized_heat_texts = []
        for t in heat_texts:
            equalized_heat_texts.append(
                equalize_row_length_in_sentence_matrix(
                    t, cfg.rapidity_rate, cfg.equalizer_params
                    )
            )
        return equalized_heat_texts
    if cfg.data_type == "pos":
        equalized_heat_files = []
        for f in source:
            equalized_heat_files.append(
                equalize_row_length_in_sentence_matrix(
                    f, cfg.rapidity_rate, cfg.equalizer_params
                    )
            )
        return equalized_heat_files
    if cfg.data_type == "lemma":
        heat_texts = []
        heat_labels = []
        for t in source:
            heat_labels.append(
                [list(' '.join([w.tagged_lemma for w in s])) for s in t]
            )
            print(heat_labels)
            heat_texts.append(
                list(
                    [transform_to_heat_sequence(
                    '_'.join([w.lemma_heat for w in s]), cfg.data_type, variation_types, cfg.rapidity_rate, cfg.clustered) for s in t]
                )
            )
        equalized_heat_texts = []
        for t in heat_texts:
            equalized_heat_texts.append(
                equalize_row_length_in_sentence_matrix(
                    t, cfg.rapidity_rate, cfg.equalizer_params
                    )
            )
        equalized_heat_labels = []
        for t in heat_labels:
            equalized_heat_labels.append(
                equalize_row_length_in_sentence_matrix(
                    t, cfg.rapidity_rate, cfg.equalizer_params, True
                    )
            )
        return (equalized_heat_texts, equalized_heat_labels)
    heat_sequences = [
        transform_to_heat_sequence(
            s, cfg.data_type, variation_types, cfg.rapidity_rate, cfg.clustered) for s in source
            ]
    equalized_heat_sequences = equalize_row_length_in_sentence_matrix(
        heat_sequences, cfg.rapidity_rate, cfg.equalizer_params
        )
    return equalized_heat_sequences