from collections import namedtuple
from itertools import product
import random
import re
from typing import Dict, List, Set, Tuple, Union
import warnings

from lxml import etree
from nltk import word_tokenize
import numpy as np
import pymorphy2


TrainingData = namedtuple('TrainingData', ['hyponyms', 'hypernyms', 'is_true'])


def load_synsets(senses_file_name: str, synsets_file_name: str) -> Dict[str, Tuple[List[tuple], tuple]]:
    with open(senses_file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    synsets = dict()
    for sense in root.getchildren():
        if sense.tag == 'sense':
            sense_id = sense.get('id').strip()
            assert len(sense_id) > 0
            synset_id = sense.get('synset_id').strip()
            assert len(synset_id) > 0
            assert sense_id.startswith(synset_id)
            term = sense.get('name').strip()
            assert len(term) > 0
            term = tuple(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip().lower(), word_tokenize(term))))
            assert len(term) > 0
            if synset_id in synsets:
                synsets[synset_id].add(term)
            else:
                synsets[synset_id] = {term}
    del xml_data, root
    with open(synsets_file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    all_synset_IDs = set()
    for synset in root.getchildren():
        if synset.tag == 'synset':
            synset_id = synset.get('id').strip()
            assert len(synset_id) > 0
            assert synset_id in synsets
            description = synset.get('definition').strip()
            if len(description) > 0:
                description = tuple(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip().lower(), word_tokenize(description))
                ))
                assert len(description) > 0
                synsets[synset_id] = (sorted(list(synsets[synset_id])), description)
            else:
                synsets[synset_id] = (sorted(list(synsets[synset_id])), tuple())
            all_synset_IDs.add(synset_id)
    assert all_synset_IDs == set(synsets.keys())
    return synsets


def tokens_from_synsets(synsets: Dict[str, Tuple[List[tuple], tuple]],
                        additional_sources: List[List[tuple]] = None) -> Dict[str, int]:
    dictionary = dict()
    word_ID = 1
    for synset_id in synsets:
        synonyms_list, description = synsets[synset_id]
        for synonym in synonyms_list:
            for token in synonym:
                if token not in dictionary:
                    dictionary[token] = word_ID
                    word_ID += 1
        for token in description:
            if token not in dictionary:
                dictionary[token] = word_ID
                word_ID += 1
    if additional_sources is not None:
        for cur_source in additional_sources:
            for term in cur_source:
                for token in term:
                    if token not in dictionary:
                        dictionary[token] = word_ID
                        word_ID += 1
    return dictionary


def load_relations(file_name: str, synsets: Dict[str, Tuple[List[tuple], tuple]],
                   relation_kind: str) -> Dict[str, List[str]]:
    assert relation_kind in {"hyponym-hypernym", "other"}
    with open(file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    relations = dict()
    for relation in root.getchildren():
        if relation.tag == 'relation':
            parent_id = relation.get('parent_id').strip()
            assert len(parent_id) > 0
            child_id = relation.get('child_id').strip()
            assert len(child_id) > 0
            relation_name = relation.get('name').strip().lower()
            assert len(relation_name) > 0
            if (parent_id in synsets) and (child_id in synsets):
                if relation_kind == "hyponym-hypernym":
                    can_continue = ((relation_name.find("hyponym") >= 0) or (relation_name.find("hypernym") >= 0))
                    if can_continue:
                        if relation_name.find("hyponym") >= 0:
                            id = child_id
                            child_id = parent_id
                            parent_id = id
                            del id
                else:
                    can_continue = ((relation_name.find("hyponym") < 0) and (relation_name.find("hypernym") < 0))
            else:
                can_continue = False
            if can_continue:
                if parent_id in relations:
                    relations[parent_id].add(child_id)
                else:
                    relations[parent_id] = {child_id}
    assert len(relations) > 0
    for parent_id in relations:
        relations[parent_id] = sorted(list(relations[parent_id]))
    return relations


def check_relations_between_hyponym_and_hypernym(relations: Dict[str, List[str]]):
    pairs = set()
    for parent_id in relations:
        for child_id in relations[parent_id]:
            pairs.add((parent_id, child_id))
    for cur_pair in pairs:
        assert (cur_pair[1], cur_pair[0]) not in pairs


def enrich_hypernyms(relations: Dict[str, List[str]]):
    enriched_relations = dict()
    for parent_id in relations:
        new_hypernyms = set()
        for child_id in relations[parent_id]:
            if child_id in relations:
                new_hypernyms |= set(relations[child_id])
        if len(new_hypernyms) > 0:
            new_hypernyms |= set(relations[parent_id])
            new_hypernyms = sorted(list(new_hypernyms))
            enriched_relations[parent_id] = new_hypernyms
            del new_hypernyms
        else:
            enriched_relations[parent_id] = relations[parent_id]
    return enriched_relations


def prepare_data_for_training(senses_file_name: str, synsets_file_name: str,
                              relations_file_name: str) -> Tuple[TrainingData, TrainingData, TrainingData]:
    synsets = load_synsets(senses_file_name=senses_file_name, synsets_file_name=synsets_file_name)
    true_relations = load_relations(file_name=relations_file_name, synsets=synsets, relation_kind="hyponym-hypernym")
    check_relations_between_hyponym_and_hypernym(true_relations)
    true_relations = enrich_hypernyms(true_relations)
    false_relations = load_relations(file_name=relations_file_name, synsets=synsets, relation_kind="other")
    synset_IDs = sorted(list(synsets.keys()))
    random.shuffle(synset_IDs)
    start_idx = 0
    end_idx = int(round(len(synset_IDs) * 0.8))
    synset_IDs_for_training = set(synset_IDs[start_idx:end_idx])
    start_idx = end_idx
    end_idx += int(round(len(synset_IDs) * 0.1))
    synset_IDs_for_validation = set(synset_IDs[start_idx:end_idx])
    start_idx = end_idx
    synset_IDs_for_testing = set(synset_IDs[start_idx:])
    del synset_IDs, start_idx, end_idx
    print('Number of synsets:')
    print('  - for training is {0};'.format(len(synset_IDs_for_training)))
    print('  - for validation is {0};'.format(len(synset_IDs_for_validation)))
    print('  - for testing is {0}.'.format(len(synset_IDs_for_testing)))
    true_relations_for_training = set()
    true_relations_for_validation = set()
    true_relations_for_testing = set()
    for parent_id in true_relations:
        if parent_id in synset_IDs_for_training:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_training, true_relations[parent_id]):
                true_relations_for_training.add((parent_id, child_id))
        elif parent_id in synset_IDs_for_validation:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_validation, true_relations[parent_id]):
                true_relations_for_validation.add((parent_id, child_id))
        elif parent_id in synset_IDs_for_testing:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_testing, true_relations[parent_id]):
                true_relations_for_testing.add((parent_id, child_id))
    print('A total number of `hyponym-hypernym` relations is {0}.'.format(
        len(true_relations_for_training) + len(true_relations_for_validation) + len(true_relations_for_testing)
    ))
    false_relations_for_training = set()
    false_relations_for_validation = set()
    false_relations_for_testing = set()
    for parent_id in false_relations:
        if parent_id in synset_IDs_for_training:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_training, false_relations[parent_id]):
                false_relations_for_training.add((parent_id, child_id))
        elif parent_id in synset_IDs_for_validation:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_validation,
                                   false_relations[parent_id]):
                false_relations_for_validation.add((parent_id, child_id))
        elif parent_id in synset_IDs_for_testing:
            for child_id in filter(lambda synset_id: synset_id in synset_IDs_for_testing, false_relations[parent_id]):
                false_relations_for_testing.add((parent_id, child_id))
    for parent_id, child_id in true_relations_for_training:
        false_relations_for_training.add((child_id, parent_id))
    for parent_id, child_id in true_relations_for_validation:
        false_relations_for_validation.add((child_id, parent_id))
    for parent_id, child_id in true_relations_for_testing:
        false_relations_for_testing.add((child_id, parent_id))
    print('A total number of other relations is {0}.'.format(
        len(false_relations_for_training) + len(false_relations_for_validation) + len(false_relations_for_testing)
    ))
    existed_relations = (true_relations_for_training | true_relations_for_validation | true_relations_for_testing |
                         false_relations_for_training | false_relations_for_validation | false_relations_for_testing)
    non_relations_for_training = set(
        filter(
            lambda it2: it2 not in existed_relations,
            filter(
                lambda it1: random.random() > 0.9998,
                product(synset_IDs_for_training, synset_IDs_for_training)
            )
        )
    )
    non_relations_for_validation = set(
        filter(
            lambda it2: it2 not in existed_relations,
            filter(
                lambda it1: random.random() > 0.9998,
                product(synset_IDs_for_validation, synset_IDs_for_validation)
            )
        )
    )
    non_relations_for_testing = set(
        filter(
            lambda it2: it2 not in existed_relations,
            filter(
                lambda it1: random.random() > 0.9998,
                product(synset_IDs_for_testing, synset_IDs_for_testing)
            )
        )
    )
    del existed_relations
    print('A total number of non-relations:')
    print('  - for training is {0};'.format(len(non_relations_for_training)))
    print('  - for validation is {0};'.format(len(non_relations_for_validation)))
    print('  - for testing is {0}.'.format(len(non_relations_for_testing)))
    false_relations_for_training |= non_relations_for_training
    false_relations_for_validation |= non_relations_for_validation
    false_relations_for_testing |= non_relations_for_testing
    hyponyms_for_training = np.array(
        [cur[0] for cur in true_relations_for_training] + [cur[0] for cur in false_relations_for_training],
        dtype=tuple
    )
    hypernyms_for_training = np.array(
        [cur[1] for cur in true_relations_for_training] + [cur[1] for cur in false_relations_for_training],
        dtype=tuple
    )
    labels_for_training = np.array(
        [1 for _ in range(len(true_relations_for_training))] + [0 for _ in range(len(false_relations_for_training))],
        dtype=np.uint8
    )
    del true_relations_for_training, false_relations_for_training
    hyponyms_for_validation = np.array(
        [cur[0] for cur in true_relations_for_validation] + [cur[0] for cur in false_relations_for_validation],
        dtype=tuple
    )
    hypernyms_for_validation = np.array(
        [cur[1] for cur in true_relations_for_validation] + [cur[1] for cur in false_relations_for_validation],
        dtype=tuple
    )
    labels_for_validation = np.array(
        [1 for _ in range(len(true_relations_for_validation))] +
        [0 for _ in range(len(false_relations_for_validation))],
        dtype=np.uint8
    )
    del true_relations_for_validation, false_relations_for_validation
    hyponyms_for_testing = np.array(
        [cur[0] for cur in true_relations_for_testing] + [cur[0] for cur in false_relations_for_testing],
        dtype=tuple
    )
    hypernyms_for_testing = np.array(
        [cur[1] for cur in true_relations_for_testing] + [cur[1] for cur in false_relations_for_testing],
        dtype=tuple
    )
    labels_for_testing = np.array(
        [1 for _ in range(len(true_relations_for_testing))] + [0 for _ in range(len(false_relations_for_testing))],
        dtype=np.uint8
    )
    del true_relations_for_testing, false_relations_for_testing
    indices = np.arange(0, labels_for_training.shape[0], 1, dtype=np.int32)
    np.random.shuffle(indices)
    data_for_training = TrainingData(
        hyponyms=hyponyms_for_training[indices],
        hypernyms=hypernyms_for_training[indices],
        is_true=labels_for_training[indices]
    )
    del hyponyms_for_training, hypernyms_for_training, labels_for_training, indices
    indices = np.arange(0, labels_for_validation.shape[0], 1, dtype=np.int32)
    np.random.shuffle(indices)
    data_for_validation = TrainingData(
        hyponyms=hyponyms_for_validation[indices],
        hypernyms=hypernyms_for_validation[indices],
        is_true=labels_for_validation[indices]
    )
    del hyponyms_for_validation, hypernyms_for_validation, labels_for_validation, indices
    indices = np.arange(0, labels_for_testing.shape[0], 1, dtype=np.int32)
    np.random.shuffle(indices)
    data_for_testing = TrainingData(
        hyponyms=hyponyms_for_testing[indices],
        hypernyms=hypernyms_for_testing[indices],
        is_true=labels_for_testing[indices]
    )
    del hyponyms_for_testing, hypernyms_for_testing, labels_for_testing, indices
    print('Data size for training is {0}.'.format(len(data_for_training.is_true)))
    print('Data size for validation is {0}.'.format(len(data_for_validation.is_true)))
    print('Data size for testing is {0}.'.format(len(data_for_testing.is_true)))
    print('')
    return data_for_training, data_for_validation, data_for_testing


def load_and_inflect_senses(senses_file_name: str, main_pos_tag: str) -> \
        Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]]:
    CASES = ["nomn", "gent", "datv", "ablt", "loct"]
    TENSES = ["past", "pres", "futr"]
    assert main_pos_tag in {"NOUN", "VERB"}
    with open(senses_file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    morph = pymorphy2.MorphAnalyzer()
    senses = dict()
    re_for_term = re.compile(r'^[\w\s\-]+$', re.U)
    n_senses = 0
    synsets_with_inflected_senses = set()
    all_synsets = set()
    for sense in root.getchildren():
        if sense.tag == 'sense':
            n_senses += 1
            sense_id = sense.get('id').strip()
            assert len(sense_id) > 0
            err_msg = 'Sense {0} has an empty synset!'.format(sense_id)
            synset_id = sense.get('synset_id').strip()
            assert len(synset_id) > 0, err_msg
            err_msg = "Sense {0} does not correspond to synset {1}!".format(sense_id, synset_id)
            assert sense_id.startswith(synset_id), err_msg
            all_synsets.add(synset_id)
            err_msg = 'Sense {0} is wrong!'.format(sense_id)
            term = sense.get('name').strip()
            assert len(term) > 0, err_msg
            term = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip().lower(), term.split())))
            assert len(term) > 0, err_msg
            search_res = re_for_term.match(' '.join(term))
            if search_res is None:
                ok = False
            else:
                ok = ((search_res.start() == 0) and (search_res.end() > search_res.start()))
            if ok:
                normal_form = sense.get('lemma').strip()
                assert len(normal_form) > 0, err_msg
                normal_form = list(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip().lower(), normal_form.split())
                ))
                assert len(normal_form) > 0, err_msg
                assert len(normal_form) == len(term), err_msg
                main_word = sense.get("main_word").strip()
                assert (len(main_word) > 0) or ((len(main_word) == 0) and (len(term) == 1)), err_msg
                if len(main_word) == 0:
                    main_word = normal_form[0]
                else:
                    main_word = list(filter(
                        lambda it2: len(it2) > 0,
                        map(lambda it1: it1.strip().lower(), main_word.split())
                    ))
                    assert len(main_word) == 1, err_msg
                    main_word = main_word[0]
                assert (' ' + ' '.join(normal_form) + ' ').find(' ' + main_word + ' ') >= 0, err_msg
                position_of_main_word = normal_form.index(main_word)
                position_of_main_word_ = position_of_main_word
                if len(term) > 1:
                    list_of_POS_tags = sense.get('poses').strip()
                    assert len(list_of_POS_tags) > 0, err_msg
                    list_of_POS_tags = list_of_POS_tags.split()
                    assert len(list_of_POS_tags) == len(term), err_msg
                else:
                    list_of_POS_tags = [sense.get('synt_type').strip()]
                    assert len(list_of_POS_tags[0]) > 0, err_msg
                if main_pos_tag == "NOUN" and (list_of_POS_tags[position_of_main_word] != "V"):
                    if list_of_POS_tags[position_of_main_word] != "N":
                        if sum(map(lambda pos: 1 if pos == "N" else 0, list_of_POS_tags)) >= 1:
                            position_of_main_word_ = list_of_POS_tags.index("N")
                        else:
                            if sum(map(lambda pos: 1 if pos == "Adj" else 0, list_of_POS_tags)) >= 1:
                                position_of_main_word_ = list_of_POS_tags.index("Adj")
                            else:
                                position_of_main_word_ = -1
                                warnings.warn("There are no main words for sense {0}.".format(sense_id))
                    if position_of_main_word_ >= 0:
                        if position_of_main_word_ != position_of_main_word:
                            parsed = parse_by_pymorphy2(term[position_of_main_word_], morph,
                                                        normal_form[position_of_main_word_])
                            if parsed is None:
                                warnings.warn('Sense {0} cannot be parsed by the PyMorphy2! '
                                              'Therefore, this sense will be skipped.'.format(sense_id))
                            elif parsed.tag.POS is not None:
                                senses[sense_id] = {
                                    noun_morphotag_to_str(parsed): tokenize_sense(term, position_of_main_word_)
                                }
                                synsets_with_inflected_senses.add(synset_id)
                            else:
                                warnings.warn('Sense {0} cannot be parsed by the PyMorphy2! '
                                              'Therefore, this sense will be skipped.'.format(sense_id))
                        else:
                            if list_of_POS_tags[position_of_main_word] == 'N':
                                noun_phrase_end = position_of_main_word + 1
                                noun_phrase_start = position_of_main_word - 1
                                while noun_phrase_start >= 0:
                                    if list_of_POS_tags[noun_phrase_start] != 'Adj':
                                        break
                                    noun_phrase_start -= 1
                                noun_phrase_start += 1
                                parsed = [parse_by_pymorphy2(term[token_idx], morph, normal_form[token_idx])
                                          for token_idx in range(noun_phrase_start, noun_phrase_end)]
                                if any(map(lambda it: (it is None) or (str(it.tag) == "UNKN"), parsed)):
                                    warnings.warn('Sense {0} cannot be parsed by the PyMorphy2! '
                                                  'Therefore, this sense will be skipped.'.format(sense_id))
                                else:
                                    variants = dict()
                                    for case in CASES:
                                        _, morpho_data = inflect_by_pymorphy2(
                                            parsed[position_of_main_word - noun_phrase_start],
                                            {case}
                                        )
                                        new_main_phrase = list(
                                            map(lambda it: inflect_by_pymorphy2(it, {case})[0], parsed))
                                        variants[noun_morphotag_to_str(morpho_data)] = tokenize_sense(
                                            tuple(term[0:noun_phrase_start] + new_main_phrase + term[noun_phrase_end:]),
                                            position_of_main_word_
                                        )
                                    senses[sense_id] = variants
                                    synsets_with_inflected_senses.add(synset_id)
                            else:
                                parsed = parse_by_pymorphy2(term[position_of_main_word], morph,
                                                            normal_form[position_of_main_word])
                                if parsed is None:
                                    warnings.warn('Sense {0} cannot be parsed by the PyMorphy2! '
                                                  'Therefore, this sense will be skipped.'.format(sense_id))
                                else:
                                    senses[sense_id] = {
                                        noun_morphotag_to_str(parsed): tokenize_sense(term, position_of_main_word_)
                                    }
                                    synsets_with_inflected_senses.add(synset_id)
                else:
                    parsed = parse_by_pymorphy2(term[position_of_main_word], morph, normal_form[position_of_main_word])
                    if parsed is None:
                        warnings.warn('Sense {0} cannot be parsed by the PyMorphy2! '
                                      'Therefore, this sense will be skipped.'.format(sense_id))
                    else:
                        variants = dict()
                        for tense in TENSES:
                            morpho_data = parsed.inflect({tense})
                            if morpho_data is not None:
                                inflected_verb = str(morpho_data.word)
                                variants[verb_morphotag_to_str(morpho_data)] = tokenize_sense(
                                    tuple(term[0:position_of_main_word] + [inflected_verb] +
                                          term[(position_of_main_word + 1):]),
                                    position_of_main_word_
                                )
                        if len(variants) > 0:
                            senses[sense_id] = variants
                            synsets_with_inflected_senses.add(synset_id)
                        else:
                            warnings.warn('Sense {0} cannot be inflected by the PyMorphy2! '
                                          'Therefore, this sense will be skipped.'.format(sense_id))
            else:
                warnings.warn('Sense {0} can contain some punctuation etc., and this is a problem. '
                              'Therefore, this sense will be skipped.'.format(sense_id))
    print('{0} words (or phrases) from {1} have been inflected.'.format(len(senses), n_senses))
    print('{0} synsets from {1} contain inflected senses.'.format(len(synsets_with_inflected_senses), len(all_synsets)))
    return senses


def parse_by_pymorphy2(source_word: str, morph: pymorphy2.MorphAnalyzer,
                       true_normal_form: str) -> pymorphy2.analyzer.Parse:
    res = None
    for it in morph.parse(source_word):
        if it.normal_form == true_normal_form:
            res = it
            break
        else:
            if it.normal_form.replace('ё', 'е') == true_normal_form.replace('ё', 'е'):
                res = it
                break
    return res


def tokenize_sense(src_tokenized: Union[tuple, list], main_word_pos: int) -> Tuple[tuple, Tuple[int, int]]:
    new_tokens = []
    main_word_start = main_word_pos
    main_word_end = main_word_start + 1
    for old_token_idx, token in enumerate(src_tokenized):
        dash_idx = token.find('-')
        if dash_idx < 0:
            subtokens = [token]
        else:
            subtokens = []
            if dash_idx > 0:
                subtokens.append(token[0:dash_idx])
            subtokens.append('-')
            token_tail = token[(dash_idx + 1):]
            dash_idx = token_tail.find('-')
            while dash_idx >= 0:
                if dash_idx > 0:
                    subtokens.append(token_tail[0:dash_idx])
                subtokens.append('-')
                token_tail = token_tail[(dash_idx + 1):]
                dash_idx = token_tail.find('-')
            if len(token_tail) > 0:
                subtokens.append(token_tail)
        new_tokens += subtokens
        if old_token_idx == main_word_pos:
            main_word_end = main_word_start + len(subtokens)
        elif old_token_idx < main_word_pos:
            main_word_start += (len(subtokens) - 1)
            main_word_end += (len(subtokens) - 1)
    return tuple(new_tokens), (main_word_start, main_word_end)


def inflect_by_pymorphy2(source_parsed_word: pymorphy2.analyzer.Parse, required_grammemes: Set[str]) -> \
        Tuple[str, pymorphy2.analyzer.Parse]:
    res = source_parsed_word.inflect(required_grammemes)
    if res is None:
        inflected = source_parsed_word
        inflected_word = source_parsed_word.word
        warnings.warn("Word `{0}` cannot be inflected.".format(inflected_word))
    else:
        inflected = res
        inflected_word = res.word
    return inflected_word, inflected


def noun_morphotag_to_str(parsed: pymorphy2.analyzer.Parse) -> str:
    if parsed.tag.POS in {"NOUN", "ADJF", "ADJS"}:
        res = str(parsed.tag.case) + "," + str(parsed.tag.gender) + "," + str(parsed.tag.number)
    else:
        res = str(parsed.tag.POS)
    return res


def verb_morphotag_to_str(parsed: pymorphy2.analyzer.Parse) -> str:
    assert parsed.tag.POS == "VERB", str(parsed.tag)
    res = str(parsed.tag.tense)
    if parsed.tag.number is not None:
        number = str(parsed.tag.number)
        if len(number) > 0:
            res += ("," + number)
    if parsed.tag.gender is not None:
        gender = str(parsed.tag.gender)
        if len(gender) > 0:
            res += ("," + gender)
    return res
