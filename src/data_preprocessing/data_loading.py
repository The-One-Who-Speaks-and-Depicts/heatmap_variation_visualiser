import datasets
import os
import re
import numpy as np
import pandas as pd
import logging

from dataclasses import dataclass

from conllu import parse
from stanza.utils.conll import CoNLL

from data_preprocessing.thematic_modelling import tag_sentences_for_topics

logger = logging.getLogger('heatmap_creator')

def load_ua_gec_data(source_directory: str) -> list[str]:
    """
    Load the Ukrainian GEC file and return the data as a list of strings.
    """
    # getting all the annotated files in the provided directory
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.ann' in f)]
    data = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            data.append([s for s in file.read().split('\n\n') if s and s.strip()])
    return [i for j in data for i in j]

def load_ua_ner_data() -> list[list[tuple]]:
    """
    Load the UA-NER and return the data as a list of list of tuples
    """
    dataset = datasets.load_dataset("benjamin/ner-uk")
    data = [i for j in [dataset['train'], dataset['validation'], dataset['test']] for i in j]
    data = [(i['tokens'], i['ner_tags']) for i in data]
    data = [list(zip(i[0], i[1])) for i in data]
    return data

def load_conllu_data_for_thematic_modelling(source_directory: str, text_separator: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Loads the .conllu files from directory
    """
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.conllu' in f)]
    data = []
    current_text = []
    current_id = ''
    sentences = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            whole_file = f.read()
            for sent in whole_file.split('\n\n'):
                if sent and sent.strip():
                    current_sent = []               
                    lines = [line for line in sent.split('\n') if line and line.strip()]
                    for line in lines:
                        if line.startswith(text_separator):
                            new_id = re.sub(text_separator, '', line).strip()
                            if not current_id:
                                current_id = new_id
                            if len(current_text) > 0 and current_text != new_id:
                                current_text = ' '.join(current_text).strip()                              
                                data.append((current_text, current_id))
                                current_text = []
                                current_id = new_id
                        if not line.startswith('#'):
                            current_text.append(line.split('\t')[1])
                            current_sent.append(line.split('\t')[1])
                    current_sent = ' '.join(current_sent).strip()
                    sentences.append((current_id, current_sent))
            # TODO: this is really bad code, because I have to repeat it just for the last sentence
            current_text = ' '.join(current_text).strip()                              
            data.append((current_text, current_id))
    df = pd.DataFrame(data, columns=['text', 'lect'])
    return (df, sentences)

def load_conllu_variation(source_directory: str) -> list[list[str]]:
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.conllu' in f)]
    data = []
    labels = []
    for f in files:
        logger.debug("Loading file %s", f)
        with open(f, "r", encoding='utf-8') as inp:
            sents = parse(inp.read())
            variation_sents = [i.metadata['variation_text'] for i in sents]
            data.append(variation_sents)
            standard_sents = [i.metadata['standard_text'] for i in sents]
            labels.append(standard_sents)
    return (data, labels) 

def load_text_by_copies(source_directory: str) -> list[list[str]]:
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.txt' in f)]
    data = []
    for f in files:
        logger.debug("Loading file %s", f)
        with open(f, "r", encoding='utf-8') as inp:
            data.append([line.strip('\n') for line in inp.readlines() if line and line.strip()])
    return data

def load_pos(source_directory: str, rapidity_rate: int) -> list[list[int]]:
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.conllu' in f)]
    data = []
    for f in files:
        doc = CoNLL.conll2doc(f)
        result_matrix = []
        for sent in doc.sentences:
            row = []
            for token in sent.tokens:
                for word in token.words:
                    misc_keys = word.misc.split('|')
                    for key in misc_keys:
                        if 'PosRapidity' in key:
                            rate = int(key.split('=')[1]) + rapidity_rate
                            heat = np.empty(len(word.text) + 2, dtype='int') 
                            heat.fill(rate)
                            row.extend(heat)
            result_matrix.append(row)
        data.append(result_matrix)
    return data

@dataclass
class LemmaForHeat:
    tagged_lemma: str = ""
    lemma_heat: str = ""


def load_lemma(source_directory: str) -> list[list[LemmaForHeat]]:
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if (os.path.isfile(os.path.join(source_directory, f)) and '.conllu' in f)]
    data = []
    for f in files:
        doc = CoNLL.conll2doc(f)
        result_matrix = []
        for sent in doc.sentences:
            row = []
            for token in sent.tokens:
                for word in token.words:
                    misc_keys = word.misc.split('|')
                    lemma = LemmaForHeat()
                    for key in misc_keys:
                        if 'LemmaErrorSpots' in key:
                            lemma.lemma_heat = key.split('=')[1]
                        if 'TaggedLemma' in key:
                            lemma.tagged_lemma = key.split('=')[1]
                    row.append(lemma)
            result_matrix.append(row)
        data.append(result_matrix)
    return data

def load_source_by_type(type, **kwargs):
    if type == "ua_gec":
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for UA-GEC mode")
        return load_ua_gec_data(kwargs.get("source_dir"))
    if type == "ua_ner":
        return load_ua_ner_data()
    if type == "automatic_thematic_modelling":
        if not kwargs.get("text_separator"):
            raise ValueError("text_separator is required for automatic_thematic_modelling")
        if not kwargs.get("lda_params"):
            raise ValueError("lda_params are obligatory for automatic_thematic_modelling")
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for UA-GEC mode")
        loaded_texts, loaded_sentences = load_conllu_data_for_thematic_modelling(kwargs.get("source_dir"), kwargs['text_separator'])
        topics = tag_sentences_for_topics(loaded_texts, loaded_sentences, kwargs.get("lda_params"))
        return topics
    if type == "text_variation":
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for text_variation mode")
        return load_text_by_copies(kwargs.get("source_dir"))
    if type == "conllu_text_variation":
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for text_variation mode")
        return load_conllu_variation(kwargs.get("source_dir"))
    if type == "pos":
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for PoS errata visualisation mode")
        if not kwargs.get("rapidity_rate"):
            raise ValueError("Rapidity rate is required for PoS errata visualisation mode")
        return load_pos(kwargs.get("source_dir"), kwargs.get("rapidity_rate"))
    if type == "lemma":
        if not kwargs.get("source_dir"):
            raise ValueError("Source directory is required for PoS errata visualisation mode")
        return load_lemma(kwargs.get("source_dir"))
    raise ValueError("Unknown source type")                    

