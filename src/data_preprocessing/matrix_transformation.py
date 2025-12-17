from dataclasses import dataclass

import logging
from copy import deepcopy
import importlib

logger = logging.getLogger('heatmap_creator')

@dataclass
class EqualizerParams:
    func_path: str = "numpy.max"
    percentile: int | None = None

def batch_sentence_to_threshold(sentence: list, threshold: int, rapidity_rate: int, label_sequence: bool = False) -> list[list]:
    if not isinstance(sentence, list):
        raise ValueError("Input should be a list")
    if len(sentence) < 1:
        raise ValueError("Input should not be empty")
    if not isinstance(rapidity_rate, int):
        raise ValueError("Rapidity rate should be an integer")
    if rapidity_rate < 1:
        raise ValueError("Rapidity rate should be a positive integer")
    if not isinstance(threshold, int):
        raise ValueError("Threshold should be an integer")
    if threshold < 1:
        raise ValueError("Threshold should be a positive integer")
    negative_rapidity = -1 * rapidity_rate
    if len(sentence) == threshold:
        return [sentence]
    if len(sentence) < threshold:
        filled_sentence = deepcopy(sentence)
        if label_sequence:
            filled_sentence.extend(' ' * (threshold - len(sentence)))
            return [filled_sentence]
        filled_sentence.extend([negative_rapidity] * (threshold - len(sentence)))
        return [filled_sentence]
    result = []
    for i in range(0, len(sentence), threshold):
        batch = sentence[i:i + threshold] if ((len(sentence) - i) > threshold) else sentence[len(sentence) - threshold:len(sentence)] 
        result.append(batch)
    return result


def equalizer_function_wrapper(
        arr: list, 
        equalizer_params: EqualizerParams = EqualizerParams()) -> int:
    mod_name, func_name = equalizer_params.func_path.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    if equalizer_params.percentile:
        logger.debug("Percentile argument of %s acquired", equalizer_params.percentile)
        return int(func(arr, equalizer_params.percentile))
    return int(func(arr))


def equalize_row_length_in_sentence_matrix(
        data: list[list], rapidity_rate: int,
        equalizer_params: EqualizerParams = EqualizerParams(),
        label_sequence: bool = False) -> list[list]:
    if not isinstance(data, list):
        raise ValueError("Input should be a list")
    if len(data) < 1:
        raise ValueError("Input should not be empty")
    if not all(isinstance(i, list) for i in data):
        raise ValueError("Input should consist of lists")
    if not all(len(i) > 0 for i in data):
        raise ValueError("Input should consist of non-empty lists")
    if not isinstance(rapidity_rate, int):
        raise ValueError("Rapidity rate should be an integer")
    if rapidity_rate < 1:
        raise ValueError("Rapidity rate should be a positive integer")
    rows_length = [len(i) for i in data]
    threshold_value = equalizer_function_wrapper(
        rows_length, equalizer_params
    )
    logger.debug("Threshold value for row length is %s", threshold_value)
    matrix_with_equalized_sentence_length = []
    for i in data:
        matrix_with_equalized_sentence_length.extend(
            batch_sentence_to_threshold(
                i, threshold_value, rapidity_rate, label_sequence
            )
        )
    return matrix_with_equalized_sentence_length