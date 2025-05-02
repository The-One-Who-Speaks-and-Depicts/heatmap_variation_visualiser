import logging
import re

from typing import Pattern

logger = logging.getLogger('heatmap_creator')

####################################################==================###############################################

def create_pattern_for_variation_type(variation_type: str) -> Pattern[str]:
    # (G/[^}]+|Spelling)
    final_pattern = re.compile("{{([^=>]+)=>([^:]+):::(variation|error)_type={}}}".format(variation_type))
    logger.info("Variation type: %s, final_pattern: %s", variation_type, final_pattern)
    return final_pattern



####################################################==================###############################################

def measure_heat_in_gec_like(input_string: str, regexps_to_clear_variation: list[Pattern[str]] = []):
    """
    Takes a sequence in GEC annotation format (usually, {error=>corrected:::error_type=TYPE};
    could also take {present=>expected:::variation_type=TYPE},
    if the object of analysis is  linguistic variation. Returns the sequence of '_'s and 'X's,
    with '_' denoting places in text with no variation/errors, and 'X' denoting places in text
    where no variation/errors, interesting for user, occurs.

    Arguments:
        input_string (str): a string, annotated in GEC format
        regexps_to_clear_variaton (list[Patterns[str]]): list of regular expressions that govern the variation/error that should be detected
    Returns:
        final_sequence (str): a sequence of '_'s and 'X's,
        with '_' denoting places in text with no variation/errors, and 'X' denoting places in text
        where no variation/errors, interesting for user, occurs.
    """
    if not isinstance(input_string, str) or not input_string.strip():
        raise ValueError("Input must be a non-empty string")

    pattern_for_others = re.compile(r'\{([^=>]*)=>([^:]+):::[^}]+\}')

    def replace_with_X(match):
        incorrect_seq_length = len(match.group(1))
        correct_seq_length = len(match.group(2))
        marker_length = incorrect_seq_length if incorrect_seq_length > 0 else correct_seq_length
        return 'X' * marker_length

    def replace_with_underscore(match):
        incorrect_word = match.group(1)
        return '_' * len(incorrect_word)

    transformed_sequence = input_string

    for pattern in regexps_to_clear_variation:
        transformed_sequence = re.sub(pattern, replace_with_X, input_string)

    transformed_sequence = re.sub(pattern_for_others, replace_with_underscore, transformed_sequence)

    # Replace all other characters with '_'
    final_sequence = re.sub(r'[^X]', '_', transformed_sequence)

    return final_sequence

def transform_binary_tagged_sequence_to_underscore_and_X(input_array):
    if not isinstance(input_array, list) or len(input_array) < 1 or not all(isinstance(i, tuple) and len(i) == 2 for i in input_array) or not all(isinstance(i[0], str) and isinstance(i[1], int) for i in input_array):
        raise ValueError("input_array must be a non-empty list of tuples, each containing a word in string format and a tag in integer format")
    output_sequence = []
    for word, tag in input_array:
        if tag == 0:
            output_sequence.append('_' * len(word))
        else:
            output_sequence.append('X' * len(word))
    return '_'.join(output_sequence)


####################################################==================###############################################

def split_string_in_half(input_string):
    if not input_string:
        raise ValueError('Input string is empty')
    length = len(input_string)

    if length == 1:
        return (input_string)

    middle = length // 2

    if length % 2 == 0:
        return input_string[:middle], input_string[middle:]
    return input_string[:middle], input_string[middle], input_string[middle+1:]


def set_indices(idx):
    if not isinstance(idx, int):
        raise ValueError('Index must be an integer')
    return idx - 2, idx - 1, idx + 1, idx + 2

def increment_by_environment(input_array, output_array, idx):
    prev_previous, previous, next, next_next = set_indices(idx)
    if prev_previous >= 0 and input_array[prev_previous] == 'X':
        output_array[idx] = output_array[idx] + 1
    if previous >= 0 and input_array[previous] == 'X':
        output_array[idx] = output_array[idx] + 2
    if next < len(input_array) and input_array[next] == 'X':
        output_array[idx] = output_array[idx] + 2
    if next_next < len(input_array) and input_array[next_next] == 'X':
        output_array[idx] = output_array[idx] + 1
    return output_array


def fill_strings_with_numbers(strings_tuple, rapidity_rate):
    if len(strings_tuple) not in [1, 2, 3]:
        raise ValueError('Tuple must contain either one, two or three strings')

    if len(strings_tuple) == 1:
        return str(rapidity_rate) + '_'

    first_string, *rest_strings = strings_tuple
    first_length = len(first_string)

    # Fill the first string with ascending numbers starting from rapidity_rate
    first_filled = [str(rapidity_rate + i) for i in range(first_length)]

    max_number = rapidity_rate + first_length - 1

    if len(rest_strings) == 1:
        second_string = rest_strings[0]
        second_length = len(second_string)

        # Fill the second string with descending numbers from max_number
        second_filled = [str(max_number - i) for i in range(second_length)]

        return ''.join(['_'.join(first_filled) + '_', '_'.join(second_filled)]) + '_'

    elif len(rest_strings) == 2:
        second_string, third_string = rest_strings
        second_length = len(second_string)
        third_length = len(third_string)

        # Fill the second string with the number that is 1 more than max_number
        second_filled = [str(max_number + 1)] * second_length

        # Fill the third string with descending numbers from max_number
        third_filled = [str(max_number - i) for i in range(third_length)]

        return ''.join(['_'.join(first_filled) + '_', '_'.join(second_filled) + '_', '_'.join(third_filled)]) + '_'

def replace_X_sequences(input_string, rapidity_rate=3):
    input_string = re.sub(r'(\d)', '\\1_', input_string)
    def replace_match(match):
        sequence_of_X = match.group(0)
        split_sequences = split_string_in_half(sequence_of_X)
        return fill_strings_with_numbers(split_sequences, rapidity_rate)

    return re.sub(r'X+', replace_match, input_string)


def transform_to_heat_sequence(
        input_data: str, input_transformation: str,
        variation_types: list[str] = [],
        rapidity_rate: int = 3, clustered: bool = False
        ):
    if rapidity_rate < 3:
        raise ValueError('Rapidity rate must be at least 3')
    input_sequence = []
    if input_transformation in ["ua_gec", "text_variation"]:
        patterns_to_detect = [create_pattern_for_variation_type(variation) for variation in variation_types]
        input_sequence = [i for i in measure_heat_in_gec_like(input_data, patterns_to_detect) if i and i.strip()]
    if input_transformation in ["ua_ner", "automatic_thematic_modelling", "stop_words"]:
        input_sequence = [i for i in transform_binary_tagged_sequence_to_underscore_and_X(input_data) if i and i.strip()]
    input_array = list(input_sequence)
    output_array = [0 for i in range(len(input_array))]
    for idx, val in enumerate(input_array):
        if val == 'X':
            if clustered:
                output_array[idx] = 'X'
            else:
                output_array[idx] = output_array[idx] + rapidity_rate
        if ((val != 'X') or ((not clustered) and (val == 'X'))):
            output_array = increment_by_environment(input_array, output_array, idx)
    if clustered:
        output_string = ''.join([str(x) for x in output_array])
        output_array = [int(i) for i in replace_X_sequences(output_string, rapidity_rate).split('_') if i != '']
    return output_array
