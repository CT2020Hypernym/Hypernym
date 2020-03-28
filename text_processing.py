"""
This module is a part of system for the automatic enrichment
of a WordNet-like taxonomy.

Copyright 2020 Ivan Bondarenko, Tatiana Batura

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import codecs
import copy
from functools import reduce
import gzip
import json
import multiprocessing
import os
import re
from typing import Dict, List, Sequence, Set, Tuple, Union

from bert.tokenization.bert_tokenization import FullTokenizer
from gensim.models.fasttext import FastText
from nltk import wordpunct_tokenize
import numpy as np
from rusenttokenize import ru_sent_tokenize
from scipy.spatial.distance import cdist
from spacy.language import Language

from trainset_preparing import calculate_sentence_matrix


def tokenize(source_text: str) -> List[str]:
    """ Prepare and tokenize a text.

    Replaces all kinds of dashes with a simple dash, tokenize a transformed text using
    the `nltk.wordpunct_tokenize` function and remove unnecessary punctuation.

    :param source_text: an input text for processing and tokenization.
    :return: a result as a Python's tuple of strings.
    """
    unicode_dashes = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2043", "&#8208;", "&#8209;",
                      "&#8210;", "&#8211;", "&#8212;", "&#8213;", "&#8259;"]
    dashes_expr = "(" + "|".join(unicode_dashes) + ")"
    re_for_dashes = re.compile(dashes_expr)
    prepared_text = re_for_dashes.sub("-", source_text)
    prepared_text = prepared_text.replace("\u2026", "...").replace("&#8230;", "...")
    prepared_text = prepared_text.replace("\u2025", "..").replace("&#8229;", "..")
    return list(filter(
        lambda it2: (len(it2) > 0) and (it2.isalnum() or (it2 in {".", ",", "-", ":", ";", "(", ")"})),
        map(lambda it1: it1.strip().lower(), wordpunct_tokenize(prepared_text))
    ))


def load_news(corpus_dir_name: str):
    """ Load the news corpus, prepared for the competition, and create a generator.

    :param corpus_dir_name: a directory with the news corpus files.
    :return: a generator for each news line.
    """
    re_for_filename = re.compile(r'^news_df_\d+.csv.gz$')
    data_files = list(map(
        lambda it2: os.path.join(os.path.normpath(corpus_dir_name), it2),
        filter(
            lambda it1: re_for_filename.match(it1.lower()) is not None,
            os.listdir(os.path.normpath(corpus_dir_name))
        )
    ))
    assert len(data_files) > 0
    for cur_file in data_files:
        with gzip.open(cur_file, mode="rt", encoding="utf-8") as fp:
            cur_line = fp.readline()
            line_idx = 1
            true_header = ["file_name", "file_sentences"]
            loaded_header = []
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    err_msg = 'File `{0}`: line {1} is wrong!'.format(cur_file, line_idx)
                    if len(loaded_header) == 0:
                        line_parts = list(filter(
                            lambda it2: len(it2) > 0,
                            map(lambda it1: it1.strip(), prep_line.split(","))
                        ))
                    else:
                        line_parts = list(filter(
                            lambda it2: len(it2) > 0,
                            map(lambda it1: it1.strip(), prep_line.split("\t"))
                        ))
                        if len(line_parts) > 2:
                            line_parts = [line_parts[0], " ".join(line_parts[1:])]
                    assert len(line_parts) == 2, err_msg
                    if len(loaded_header) == 0:
                        assert line_parts == true_header, err_msg
                        loaded_header = line_parts
                    else:
                        url = line_parts[0].lower()
                        text_list = line_parts[1]
                        assert url.endswith(".htm") or url.endswith(".html"), err_msg
                        try:
                            data = json.loads(text_list, encoding='utf-8')
                        except:
                            data = None
                        assert data is not None, err_msg
                        assert isinstance(data, list), err_msg
                        assert len(data) > 0, err_msg
                        for text in data:
                            yield text
                cur_line = fp.readline()
                line_idx += 1


def load_wiki(file_name: str):
    """ Load the Wikipedia dump, tokenize all texts from this dump by sentences and create a generator.

    :param file_name:
    :return: a generator for each news line (all such lines are prepared and tokenized by sentences).
    """
    with gzip.open(file_name, mode="rt", encoding="utf-8") as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                yield from filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), ru_sent_tokenize(prep_line)))
            cur_line = fp.readline()


def prepare_senses_index_for_search(senses_dict: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]]) -> \
        Dict[str, Set[str]]:
    """ Build a search index for a fast selection of sentence candidates, which contain some sense from the RuWordNet.

    The RuWordNet contains a lot of terms (senses in the RuWordNet terminology), and if we want to find possible
    occurrences in each input sentence using the exhaustive search, then we will do it very-very long, with
    time complexity is O(n). So, we can divide the search procedure into two steps:

    1) we select a sub-set of all RuWordNet's terms, which potentially can be a part of some sentence, using
    a hash table of single words from all terms, and we do it with the constant time complexity O(1), because
    it is the search complexity in the hash table;

    2) we apply a full linear search for the selected sub-set of terms instead of all RuWordNet's terms.

    And this function needs for building such search index in a form of the hash table (i.e., the Python's dictionary),
    where keys are single words of the RuWordNet terms, and values are sense IDs of terms with these words.

    :param senses_dict: a dictionary with inflected terms (see `ruwordnet_parsing.load_and_inflect_senses` function).
    :return: the created search index.
    """
    index = dict()
    for sense_id in senses_dict:
        for morpho_tag in senses_dict[sense_id]:
            tokens = senses_dict[sense_id][morpho_tag][0]
            main_word_start, main_word_end = senses_dict[sense_id][morpho_tag][1]
            for main_token in filter(lambda it: it.isalnum(), tokens[main_word_start:main_word_end]):
                if main_token in index:
                    index[main_token].add(sense_id)
                else:
                    index[main_token] = {sense_id}
    return index


def startswith(full_text: tuple, subphrase: tuple) -> int:
    """ Check that the specified text starts with the specified subphrase without considering of punctuation.

    Text and subphrase are tokenized, i.e. they are tuples of strings. Matching is realized recursively.

    :param full_text: a tokenized text (tuple of strings).
    :param subphrase: a tokenized subphrase (tuple of strings).
    :return: a number of text's words, which coincide with all subphrase's words.
    """
    n_full = len(full_text)
    n_sub = len(subphrase)
    if (n_sub == 0) or (n_full == 0):
        return 0
    if n_sub > n_full:
        return 0
    if ' '.join(full_text) == ' '.join(subphrase):
        return n_full
    if ' '.join(full_text[0:n_sub]) == ' '.join(subphrase):
        return n_sub
    if full_text[0].isalnum() and subphrase[0].isalnum():
        if full_text[0] != subphrase[0]:
            return 0
        res = startswith(full_text[1:], subphrase[1:])
        if res == 0:
            return 0
        return res + 1
    if (not full_text[0].isalnum()) and (not subphrase[0].isalnum()):
        if (n_full < 2) or (n_sub < 2):
            return 0
        res = startswith(full_text[1:], subphrase[1:])
        if res == 0:
            return 0
        return res + 1
    if full_text[0].isalnum():
        return startswith(full_text, subphrase[1:])
    res = startswith(full_text[1:], subphrase)
    if res == 0:
        return 0
    return res + 1


def find_subphrase(full_text: tuple, subphrase: tuple) -> Union[Tuple[int, int], None]:
    """ Find bounds of the specified subphrase in the specified text without considering of punctuation.

    For example, if we want to find bounds of the subphrase ("hello", "how", "are", "you") in the text ("oh", ",",
    "hello", "!", "how", "are", "you", "doing", "today", "?"), then we expect the result (2, 7). The same result will be
    expected, if the above-mentioned subphrase contains a comma between "hello" and "how", for example. But if the text
    will be modified in the following way, e.g. ("oh", ",", "hello", "how", "are", "you", "doing", "today", "?"), then
    we expect another result (2, 6).

    :param full_text: a tokenized text (tuple of strings).
    :param subphrase: a tokenized subphrase (tuple of strings).
    :return: a two-element tuple, i.e. bounds of subphrase, if these bounds are found, and None in another case.
    """
    assert subphrase[0].isalnum() and subphrase[-1].isalnum(), \
        "The subphrase `{0}` is wrong! Any subphrase must be started and ended with alphabetic or " \
        "numeric words!".format(' '.join(subphrase))
    n_full = len(full_text)
    n_sub = len(subphrase)
    if n_sub > n_full:
        return None
    start_pos = -1
    end_pos = -1
    for token_idx in range(n_full):
        if full_text[token_idx] == subphrase[0]:
            n = startswith(full_text[token_idx:], subphrase)
            if n > 0:
                start_pos = token_idx
                end_pos = start_pos + n
                break
    if start_pos >= 0:
        return start_pos, end_pos
    return None


def join_tokens(tokens: Sequence[str]) -> str:
    joined = ''.join(tokens)
    joined_new = joined.replace(' ', '')
    while joined_new != joined:
        joined = joined_new
        joined_new = joined.replace(' ', '')
    return joined_new


def postprocess_pos_tags(list_of_tokens: List[Sequence[str]], list_of_pos_tags: List[Sequence[str]],
                         tokenized_texts: Sequence[tuple]) -> List[tuple]:
    list_of_prepared_pos_tags = []
    for tokens, pos_tags, tokenized_text in zip(list_of_tokens, list_of_pos_tags, tokenized_texts):
        idx1 = 0
        idx2 = 0
        indices_map = []
        err_msg = '`{0}` != `{1}`'.format(tokenized_text, tokens)
        while (idx1 < len(tokenized_text)) and (idx2 < len(tokens)):
            if tokenized_text[idx1] == tokens[idx2]:
                indices_map.append(((idx1,), (idx2,)))
                idx1 += 1
                idx2 += 1
            else:
                best_indinces_comb = None
                for other_idx1 in range(idx1 + 1, len(tokenized_text) + 1):
                    for other_idx2 in range(idx2 + 1, len(tokens) + 1):
                        if join_tokens(tokenized_text[idx1:other_idx1]) == join_tokens(tokens[idx2:other_idx2]):
                            if best_indinces_comb is None:
                                best_indinces_comb = (tuple(range(idx1, other_idx1)), tuple(range(idx2, other_idx2)))
                            else:
                                if max(other_idx1 - idx1, other_idx2 - idx2) < \
                                        max(len(best_indinces_comb[0]), len(best_indinces_comb[1])):
                                    best_indinces_comb = (
                                        tuple(range(idx1, other_idx1)), tuple(range(idx2, other_idx2)))
                assert best_indinces_comb is not None, err_msg + ', {0}'.format(indices_map)
                indices_map.append(best_indinces_comb)
                idx1 = best_indinces_comb[0][-1] + 1
                idx2 = best_indinces_comb[1][-1] + 1
        assert (idx1 >= len(tokenized_text)) and (idx2 >= len(tokens)), err_msg
        prepared_pos_tags = []
        for indices_of_tokens, indices_of_pos_tags in indices_map:
            if len(indices_of_pos_tags) == 1:
                for _ in range(len(indices_of_tokens)):
                    prepared_pos_tags.append(pos_tags[indices_of_pos_tags[0]])
            else:
                pos_freq = dict()
                for idx in indices_of_pos_tags:
                    pos_freq[pos_tags[idx]] = pos_freq.get(pos_tags[idx], 0) + 1
                pos_freq_ = sorted([(pos, pos_freq[pos]) for pos in pos_freq.keys()], key=lambda it: (-it[1], it[0]))
                del pos_freq
                for _ in range(len(indices_of_tokens)):
                    prepared_pos_tags.append(pos_freq_[0][0])
                del pos_freq_
        assert len(prepared_pos_tags) == len(tokenized_text), err_msg
        del indices_map
        list_of_prepared_pos_tags.append(tuple(prepared_pos_tags))
    return list_of_prepared_pos_tags


def calculate_pos_tags(tokenized_texts: Sequence[tuple], udpipe_pipeline: Language,
                       pool: Union[multiprocessing.Pool, None] = None) -> List[tuple]:
    list_of_tokens = []
    list_of_pos_tags = []
    n_threads = (1 if pool is None else getattr(pool, "_processes"))
    for doc in udpipe_pipeline.pipe([' '.join(cur) for cur in tokenized_texts]):
        list_of_tokens.append(tuple([token.text for token in doc]))
        list_of_pos_tags.append(tuple([token.pos_.lower() for token in doc]))
    if n_threads > 1:
        n_data_parts = int(np.ceil(len(tokenized_texts) / float(n_threads)))
        parts_of_buffer = [
            (
                list_of_tokens[(idx * n_data_parts):((idx + 1) * n_data_parts)],
                list_of_pos_tags[(idx * n_data_parts):((idx + 1) * n_data_parts)],
                tokenized_texts[(idx * n_data_parts):((idx + 1) * n_data_parts)]
            )
            for idx in range(n_threads - 1)
        ]
        parts_of_buffer.append(
            (
                list_of_tokens[((n_threads - 1) * n_data_parts):],
                list_of_pos_tags[((n_threads - 1) * n_data_parts):],
                tokenized_texts[((n_threads - 1) * n_data_parts):]
            )
        )
        parts_of_result = list(pool.starmap(postprocess_pos_tags, parts_of_buffer))
        list_of_prepared_pos_tags = parts_of_result[0]
        for cur in parts_of_result[1:]:
            list_of_prepared_pos_tags += cur
        del parts_of_buffer, parts_of_result
    else:
        list_of_prepared_pos_tags = postprocess_pos_tags(list_of_tokens, list_of_pos_tags, tokenized_texts)
    return list_of_prepared_pos_tags


def find_senses_in_texts(tokenized_texts: Sequence[tuple], pos_tags_for_these_texts: Sequence[tuple],
                         senses_dict: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                         search_index_for_senses: Dict[str, Set[str]], main_pos_tag: str) -> \
        List[Union[Dict[str, Dict[str, Tuple[int, int]]], None]]:
    """ Analyze an input sentence and find all admissible occurrences of the RuWordNet's terms (senses).

    :param tokenized_texts: a sequence of input sentences, which are tokenized using the `tokenize` function.
    :param pos_tags_for_these_texts: a sequence of part-of-speech tags, corresponded to tokens of input sentences.
    :param senses_dict: a dictionary with inflected terms (see `ruwordnet_parsing.load_and_inflect_senses` function).
    :param search_index_for_senses: a search index, which is built using the `prepare_senses_index_for_search` function.
    :param main_pos_tag: target POS-tag (part of speech) all found terms (senses).
    :return: None or the Python's dictionary "sense ID" -> "morphotag" -> "bounds in the sentence"
    """
    assert main_pos_tag.lower() in {'verb', 'noun'}, '`{0}` is inadmissible part of speech!'.format(main_pos_tag)
    total_founds = []
    for tokenized_text, pos_tags in zip(tokenized_texts, pos_tags_for_these_texts):
        filtered_sense_IDs = set()
        for token in tokenized_text:
            if token.isalnum():
                filtered_sense_IDs |= search_index_for_senses.get(token, set())
        if len(filtered_sense_IDs) == 0:
            total_founds.append(None)
        else:
            res = dict()
            for sense_ID in filtered_sense_IDs:
                founds = dict()
                for morpho_tag in senses_dict[sense_ID]:
                    sense_tokens = senses_dict[sense_ID][morpho_tag][0]
                    sense_bounds = find_subphrase(full_text=tokenized_text, subphrase=sense_tokens)
                    if sense_bounds is not None:
                        if (sense_bounds[1] - sense_bounds[0]) > 1:
                            founds[morpho_tag] = sense_bounds
                        else:
                            if main_pos_tag.lower() == pos_tags[sense_bounds[0]]:
                                founds[morpho_tag] = sense_bounds
                if len(founds) > 0:
                    res[sense_ID] = founds
                del founds
            if len(res) == 0:
                total_founds.append(None)
            else:
                total_founds.append(res)
    return total_founds


def tokenize_with_BERT(source_text: Sequence[str], term_bounds: Tuple[int, int],
                       bert_tokenizer: FullTokenizer) -> Union[Tuple[str, Tuple[int, int]], None]:
    assert len(term_bounds) == 2
    assert (term_bounds[0] >= 0) and (term_bounds[0] < len(source_text))
    assert (term_bounds[1] > 0) and (term_bounds[0] <= len(source_text))
    assert term_bounds[0] < term_bounds[1]
    bpe = []
    term_start = -1
    term_end = -1
    for token_idx, token_text in enumerate(source_text):
        bpe_of_token = bert_tokenizer.tokenize(token_text)
        err_msg = 'The text `{0}` cannot be tokenized for BERT!'.format(' '.join(source_text))
        assert len(bpe_of_token) > 0, err_msg
        assert (bpe_of_token[0] != '[CLS]') and (bpe_of_token[-1] != '[SEP]'), err_msg
        if (token_idx != term_bounds[0]) and (token_idx != term_bounds[1]):
            bpe += bpe_of_token
        elif token_idx == term_bounds[0]:
            term_start = len(bpe)
            bpe += bpe_of_token
        else:
            term_end = len(bpe)
            bpe += bpe_of_token
    if (len(bpe) == 0) or (len(bpe) > 31):
        return None
    return ' '.join(bpe), (term_start, term_end)


def tokenize_many_texts(source_texts: List[str], min_sentence_length: int, max_sentence_length: int) -> List[tuple]:
    tokenized_texts = []
    for new_text in source_texts:
        new_tokenized_text = tuple(filter(
            lambda it2: len(it2) > 0,
            map(
                lambda it1: it1.strip(),
                reduce(lambda x, y: x + tokenize(y), new_text.split(), [])
            )
        ))
        if not ((len(new_tokenized_text) < min_sentence_length) or (len(new_tokenized_text) > max_sentence_length)):
            tokenized_texts.append(new_tokenized_text)
    return tokenized_texts


def calculate_sense_occurrences_in_texts(source_texts: List[str],
                                         senses_dict: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                                         search_index_for_senses: Dict[str, Set[str]], n_sentences_per_morpho: int,
                                         min_sentence_length: int, max_sentence_length: int,
                                         bert_tokenizer: FullTokenizer, udpipe_pipeline: Language, main_pos_tag: str,
                                         homonyms: Union[Dict[str, Tuple[np.ndarray, str]], None] = None,
                                         fasttext_model: Union[FastText, None] = None,
                                         pool: Union[multiprocessing.Pool, None] = None) -> \
        Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]]:
    """ Create a dictionary of occurrences of the RuWordNet's terms in the collection of input texts.

    Found occurrences are represented as a Python dictionary, where the key is a term (sense) ID, and the value is
    another Python's dictionary with a list of texts and occurrence bounds of the term in these texts
    (dictionary's values) by term's morphological tags (dictionary's keys). Each occurrence information item consists of
    a string representation of context, where all tokens are separated by spaces, and a two-element tuple with
    occurrence bounds in this context (positions of start token and of next after last one).

    Some senses can be homonyms, i.e. they have the same way of writing, but their semantics are different.
    In this case, we have to select an occurrence with the correct context, which is corresponded to one of
    the possible semantic values. For solving this problem we use a special dictionary of homonyms, which was built
    using the RuWordNet data, and a FastText model to calculate semantic similarity.

    :param source_texts: a list of input texts for tokenization and term search.
    :param senses_dict: a dictionary with inflected terms (see `ruwordnet_parsing.load_and_inflect_senses` function).
    :param search_index_for_senses: a search index, which is built using the `prepare_senses_index_for_search` function.
    :param n_sentences_per_morpho: a maximal number of sentences with term occurrences per single morphological tag.
    :param min_sentence_length: a minimal number of tokens in the sentence.
    :param max_sentence_length: a maximal number of tokens in the sentence.
    :param all_occurrences: all found occurrences (before initial of this function it must be an empty list).
    :param bert_tokenizer: a tokenizer for BERT model.
    :param udpipe_pipeline: a Spacy-UDPipe model for tokenization and morphological analysis.
    :param main_pos_tag: target POS-tag (part of speech) all found terms (senses).
    :param homonyms: dictionary with all homonyms from the RuWordNet and their FastText representations.
    :param fasttext_model: a FastText model for calculation of text representation.
    :param pool: a pool of worker processes (can be None for suppression of parallelization).
    :return: we don't return any object, but we re-write the `all_occurrences`.
    """
    assert ((homonyms is None) and (fasttext_model is None)) or \
           ((homonyms is not None) and (fasttext_model is not None))
    all_occurrences = dict()
    if pool is None:
        n_processes = 1
    else:
        n_processes = getattr(pool, "_processes")
    if n_processes > 1:
        n_data_parts = int(np.ceil(len(source_texts) / float(n_processes)))
        parts_of_buffer = [(source_texts[(idx * n_data_parts):((idx + 1) * n_data_parts)],
                            min_sentence_length, max_sentence_length)
                           for idx in range(n_processes - 1)]
        parts_of_buffer.append((source_texts[((n_processes - 1) * n_data_parts):],
                                min_sentence_length, max_sentence_length))
        parts_of_result = list(pool.starmap(tokenize_many_texts, parts_of_buffer))
        tokenized_texts = parts_of_result[0]
        for cur in parts_of_result[1:]:
            tokenized_texts += cur
        del parts_of_buffer, parts_of_result
        pos_tags = calculate_pos_tags(tokenized_texts, udpipe_pipeline, pool)
        parts_of_buffer = [(tokenized_texts[(idx * n_data_parts):((idx + 1) * n_data_parts)],
                            pos_tags[(idx * n_data_parts):((idx + 1) * n_data_parts)],
                            senses_dict, search_index_for_senses, main_pos_tag)
                           for idx in range(n_processes - 1)]
        parts_of_buffer.append((tokenized_texts[((n_processes - 1) * n_data_parts):],
                                pos_tags[((n_processes - 1) * n_data_parts):],
                                senses_dict, search_index_for_senses, main_pos_tag))
        parts_of_result = list(pool.starmap(find_senses_in_texts, parts_of_buffer))
        founds_in_all_texts = parts_of_result[0]
        for cur in parts_of_result[1:]:
            founds_in_all_texts += cur
    else:
        tokenized_texts = tokenize_many_texts(source_texts, min_sentence_length, max_sentence_length)
        pos_tags = calculate_pos_tags(tokenized_texts, udpipe_pipeline, pool)
        founds_in_all_texts = find_senses_in_texts(tokenized_texts=tokenized_texts, pos_tags_for_these_texts=pos_tags,
                                                   senses_dict=senses_dict,
                                                   search_index_for_senses=search_index_for_senses,
                                                   main_pos_tag=main_pos_tag)
    for new_tokenized_text, founds in zip(tokenized_texts, founds_in_all_texts):
        if founds is not None:
            if homonyms is None:
                filtered_founds = founds
            else:
                if len(set(founds.keys()) & set(homonyms.keys())) > 0:
                    bounds_of_homonyms = dict()
                    for sense_id in set(founds.keys()) & set(homonyms.keys()):
                        for morpho_tag in founds[sense_id]:
                            sense_bounds = founds[sense_id][morpho_tag]
                            if sense_bounds in bounds_of_homonyms:
                                bounds_of_homonyms[sense_bounds].add(sense_id)
                            else:
                                bounds_of_homonyms[sense_bounds] = {sense_id}
                    bounds_of_homonyms = dict(map(
                        lambda it2: (it2, bounds_of_homonyms[it2]),
                        filter(
                            lambda it1: len(bounds_of_homonyms[it1]) > 1,
                            bounds_of_homonyms
                        )
                    ))
                    if len(bounds_of_homonyms) == 0:
                        filtered_founds = founds
                    else:
                        matrix_of_text = calculate_sentence_matrix(sentence=new_tokenized_text,
                                                                   fasttext_model=fasttext_model)
                        filtered_founds = dict()
                        for sense_bounds in bounds_of_homonyms:
                            homonym_IDs = sorted(list(bounds_of_homonyms[sense_bounds]))
                            best_sense_id = homonym_IDs[0]
                            sense_matrix = homonyms[best_sense_id][0]
                            distances = cdist(sense_matrix, matrix_of_text, metric='cosine')
                            best_distance = np.mean(np.max(distances, axis=1))
                            del sense_matrix, distances
                            for sense_id in homonym_IDs[1:]:
                                sense_matrix = homonyms[sense_id][0]
                                distances = cdist(sense_matrix, matrix_of_text, metric='cosine')
                                cur_distance = np.mean(np.max(distances, axis=1))
                                del sense_matrix, distances
                                if cur_distance < best_distance:
                                    best_distance = cur_distance
                                    best_sense_id = sense_id
                            filtered_founds[best_sense_id] = founds[best_sense_id]
                        del matrix_of_text
                    del bounds_of_homonyms
                else:
                    filtered_founds = founds
            del founds
            for sense_id in filtered_founds:
                if sense_id not in all_occurrences:
                    all_occurrences[sense_id] = dict()
                for morpho_tag in filtered_founds[sense_id]:
                    if morpho_tag not in all_occurrences[sense_id]:
                        all_occurrences[sense_id][morpho_tag] = []
                    sense_bounds = filtered_founds[sense_id][morpho_tag]
                    new_item = tokenize_with_BERT(new_tokenized_text, sense_bounds, bert_tokenizer)
                    if new_item is not None:
                        if new_item not in all_occurrences[sense_id][morpho_tag]:
                            if len(all_occurrences[sense_id][morpho_tag]) < n_sentences_per_morpho:
                                all_occurrences[sense_id][morpho_tag].append(new_item)
                                all_occurrences[sense_id][morpho_tag].sort(key=lambda val: len(val[0]))
                            else:
                                if len(new_tokenized_text) < len(all_occurrences[sense_id][morpho_tag][0][0]):
                                    old_items = all_occurrences[sense_id][morpho_tag][1:]
                                    all_occurrences[sense_id][morpho_tag] = [new_item] + old_items
    return all_occurrences


def join_sense_occurrences_in_texts(all_occurrences: List[Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]]],
                                    n_sentences_per_morpho: int) -> \
        Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]]:
    """ Join some dictionaries with information about occurrences of terms (senses) in various texts.

    :param all_occurrences: list of dictionaries that were built using the `calculate_sense_occurrences_in_texts`.
    :param n_sentences_per_morpho: a maximal number of sentences with term occurrences per single morphological tag.
    :return: a united dictionary like the one returned by the `calculate_sense_occurrences_in_texts`.
    """
    res = copy.deepcopy(all_occurrences[0])
    for cur in all_occurrences[1:]:
        for sense_id in cur:
            if sense_id in res:
                for morpho_tag in cur[sense_id]:
                    if morpho_tag in res[sense_id]:
                        res[sense_id][morpho_tag] += cur[sense_id][morpho_tag]
                        res[sense_id][morpho_tag] = sorted(list(set(res[sense_id][morpho_tag])),
                                                           key=lambda it: (it[0], it[1][0], it[1][1]))
                    else:
                        res[sense_id][morpho_tag] = cur[sense_id][morpho_tag]
            else:
                res[sense_id] = cur[sense_id]
    for sense_id in res:
        for morpho_tag in res[sense_id]:
            if len(res[sense_id][morpho_tag]) > n_sentences_per_morpho:
                res[sense_id][morpho_tag] = res[sense_id][morpho_tag][0:n_sentences_per_morpho]
    return res


def load_sense_occurrences_in_texts(file_name: str) -> Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]]:
    with codecs.open(filename=file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        all_occurrences_of_senses = json.load(fp)
    for sense_id in all_occurrences_of_senses:
        for morpho_tag in all_occurrences_of_senses[sense_id]:
            all_occurrences_of_senses[sense_id][morpho_tag] = [
                (text.lower(), tuple(occurrence_bounds))
                for text, occurrence_bounds in all_occurrences_of_senses[sense_id][morpho_tag]
            ]
    return all_occurrences_of_senses
