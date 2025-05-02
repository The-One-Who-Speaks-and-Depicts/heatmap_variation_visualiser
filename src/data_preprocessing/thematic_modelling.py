import logging

import pandas as pd

from corpus_distance import cdutils
from corpus_distance.data_preprocessing import topic_modelling as tm

logger = logging.getLogger('heatmap_creator')


def tag_sentences_for_topics(
        df: pd.DataFrame, sentences: list[tuple[str, str]],
        lda_params: tm.LDAParams = tm.LDAParams()) -> list[list[tuple]]:
    lects = cdutils.get_lects_from_dataframe(df)
    logger.debug("Lects detected, number of lects is %s", len(lects))
    logger.debug("Starting topic modelling")
    lects_with_topics = tm.get_topic_words_for_lects(df, lects, lda_params)
    tagged_sentences = []
    for sentence in sentences:
        topic_words = lects_with_topics.get(sentence[0])
        tagged_sentence = [(w, 1 if w in topic_words else 0) for w in sentence[1].split()]
        logger.debug("Tagged sentence is %s", tagged_sentence)
        tagged_sentences.append(tagged_sentence)
    logger.debug("Topic modelling finished")
    return tagged_sentences