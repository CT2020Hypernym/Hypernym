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
    """ Load all synsets from the RuWordNet

    All loaded synsets are presented as a Python dictionary (dict). Any synset is specified by its string ID,
    for example, "147272-N". This ID is a key to a value in the created dictionary. A value of this dictionary
    consists of synset definition (if this definition is not empty) and a list of all synonyms (senses) in
    corresponded synset. Texts of synonyms (senses) and synset definitions are lowercased and tokenized with
    the nltk.word_tokenize, and each text is a Python's tuple of strings.

    :param senses_file_name: the RuWordNet's XML file with senses (for example, "senses.N.xml" for nouns)
    :param synsets_file_name: the RuWordNet's XML file with synsets (for example, "synsets.N.xml" for nouns)
    :return: an above-described dictionary with information about synsets.
    """
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
    """ Generate a vocabulary of all possible tokens from the RuWordNet's synsets and additional texts.

    :param synsets: a synsets dictionary created by the `load_synsets` function.
    :param additional_sources: an optional list of additional text corpora with lowercased and tokenized texts.
    :return: a Python's dictionary "word text" - "word ID" (all IDs are unique positive integers).
    """
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
    """ Load semantic relations between synsets.

    Loaded relations between synsets are directional from start synset to end one (for the "hyponym-hypernym" relation
    a start synset is a hyponym and an end synset is a hypernym). These relations are presented as a Python's
    dictionary, where each key corresponds to start synset in relations, and the key's value is a list of end synsets.
    All synsets are specified by their IDs, such as "147272-N" or something of the sort.


    :param file_name: the RuWordNet's XML file with relations (for example, "synset_relations.N.xml" for nouns).
    :param synsets: a synsets dictionary created by the `load_synsets` function.
    :param relation_kind: "hyponym-hypernym" or "other".
    :return: an above-described dictionary with relations between synsets.
    """
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
    """
    Check the "hyponym-hypernym" relations between synsets and raise the AssertionError in case of bug (for debugging).
    :param relations: a relations dictionary created by the `load_relations` function.
    """
    pairs = set()
    for parent_id in relations:
        for child_id in relations[parent_id]:
            pairs.add((parent_id, child_id))
    for cur_pair in pairs:
        assert (cur_pair[1], cur_pair[0]) not in pairs


def enrich_hypernyms(relations: Dict[str, List[str]]):
    """
    Enrich the "hyponym-hypernym" relations, i.e. add hypernyms of the second order.
    :param relations: a relations dictionary created by the `load_relations` function. This dictionary will be modified.
    """
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
    """ Prepare data for training, validation, and testing of a term-based hypernym classifier.

    This function prepares three disjoint datasets for building of a term-based hypernym classifier, i.e. which
    classifies pairs of terms as hyponyms and hypernyms (or not hypernyms) without any context of these terms usage.

    :param senses_file_name: the RuWordNet's XML file with senses (for example, "senses.N.xml" for nouns).
    :param synsets_file_name: the RuWordNet's XML file with synsets (for example, "synsets.N.xml" for nouns).
    :param relations_file_name: the RuWordNet's XML file with relations (e.g., "synset_relations.N.xml" for nouns).
    :return: Three namedtuples (for training, for validation, and for testing, accordingly) of the TrainingData type.
    """
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
    """ Load all terms (senses) of a target kind (nouns or verbs) from the RuWordNet and inflect theirs by morphology.

    Any term (sense, in the RuWordNet terminology) can be a single word (for example, "УЛЫБКА") or a multi-word phrase
    with own syntactic structure (for example, "ТЮРЬМА ДЛЯ ОСОБО ОПАСНЫХ ПРЕСТУПНИКОВ"). And such term can be used in
    many texts from Wikipedia, various newspapers, fiction, etc. in different linguistic forms (inflections) depending
    on syntactic function of this term in the sentence. So, for fast search of the term in texts we can do one of two
    ways:

    1) lemmatize each analyzed text, whereupon search the term's lemmas (they are specified in the RuWordNet) in
    text's lemmas;
    2) inflect the term from the RuWordNet by several grammatical categories, whereupon search the term's inflections
    (or declensions, in other words) in source texts.

    The second way is faster than the first, because we have millions of texts from external collections and a few tens
    of thousands of terms from the RuWordNet. Besides, morphological analysis of some unknown text is a more difficult
    thing in comparison with morphological analysis of the RuWordNet's term, since the RuWordNet contains an important
    part of each its terms morphology.

    So, this function loads all terms from the RuWordNet by their part of speech (nouns or verbs only) and apply
    a declension procedure by cases (for nouns) or by tenses (for verbs). For multi-word terms, i.e. which are phrases,
    we know a main word and parts of speech for all words in the phrase (this information is specified in
    the RuWordNet), and such knowledge helps us to detect a syntactic structure of the term through simple heuristics
    without full syntactic parsing. And if we have syntactic information about the multi-word term, then we can inflect
    it correctly in a morphological sense.

    The above-described heuristics for syntactic parsing are:

    1) if the main word is a noun, then we find all neighboring adjectives before this word, and we consider these
    adjectives and the main noun as a noun chunk (in the terminology of the categorial grammar) and inflect words of
    this chunk only, without changing other words in the analyzed term;

    2) if the main word is a verb, then we inflect this verb only.

    Nouns (and related adjectives) are inflected by cases, and verbs are inflected by tenses.

    Results of this function are presented as a Python's dictionary. A string sense ID is used as a key in this
    dictionary, and a key's value is another Python's dictionary, in which key is a brief grammatical description
    (e.g., "gent-masc-plur" for noun which has a genitive case, a masculine gender, and a plural number)
    and value is a concrete morphological form of the term with bounds of the main word. A small possible example of
    the result is shown below:

    {
        "125142-N-169771": {
            "nomn,masc,sing": (('северный', 'флот'), (1, 2)),
            "gent,masc,sing": (('северного', 'флота'), (1, 2)),
            "datv,masc,sing": (('северному', 'флоту'), (1, 2)),
            "ablt,masc,sing": (('северным', 'флотом'), (1, 2)),
            "loct,masc,sing": (('северном', 'флоте'), (1, 2))
        },
        "9923-N-123297": {
            "nomn,masc,sing": (('город', 'федерального', 'значения'), (0, 1)),
            "gent,masc,sing": (('города', 'федерального', 'значения'), (0, 1)),
            "datv,masc,sing": (('городу', 'федерального', 'значения'), (0, 1)),
            "ablt,masc,sing": (('городом', 'федерального', 'значения'), (0, 1)),
            "loct,masc,sing": (('городе', 'федерального', 'значения'), (0, 1))
        },
        "106216-N-131944": {
            "nomn,femn,sing": (('чукча',), (0, 1)),
            "gent,femn,sing": (('чукчи',), (0, 1)),
            "datv,femn,sing": (('чукче',), (0, 1)),
            "ablt,femn,sing": (('чукчей',), (0, 1)),
            "loct,femn,sing": (('чукче',), (0, 1))
        }
    }

    :param senses_file_name: the RuWordNet's XML file with senses (for example, "senses.N.xml" for nouns).
    :param main_pos_tag: target kind (part of speech) for the RuWordNet's terms (senses).
    :return: an above-described dictionary with inflected terms (senses).
    """
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
    """ Find a true variant of morphological parsing using the PyMorphy library.

    The PyMorphy is a good library for the morphological analysis, and it is very fast, but it cannot solve
    a morphological homonymy, returning all possible variants of morpho-parsing. But we know a normal form of
    analyzed wordform, because the normal form is determined in the RuWordNet, and we can use this information
    to select a true variant among PyMorphy's results.

    :param source_word: text of source word for morphological parsing.
    :param morph: the PyMorphy analyzer.
    :param true_normal_form: a normal form of the source word, which is known from the RuWordNet.
    :return: a parsed result with a true variant of morphological analysis.
    """
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
    """ Do additional tokenization of all term's words.

    All terms of the RuWordNet are tokenized using spaces (e.g., "ПРЕДСТАВИТЕЛЬСТВО ЗА ГРАНИЦЕЙ"), but some terms can
    contain punctuation, such as dash and comma (for example, "ФИЛОСОФ-ПРАГМАТИК" or "ПРЕСТУПЛЕНИЕ ПРОТИВ СВОБОДЫ, ЧЕСТИ
    И ДОСТОИНСТВА"). We skip all terms with the comma, because morphological parsing of such terms using simple
    heuristics is difficult. But we process terms with the dash in the following way: we concert the dash as a separate
    token. Also, we correct the position of the main word in the term accordingly with new re-tokenization. Besides,
    if the main word contained the dash before re-tokenization, then after re-tokenization this word can consist of
    several words. So, we use bounds of the main phrase instead of the single main word position.

    :param src_tokenized: source tokenized term.
    :param main_word_pos: a position of the main token in the term.
    :return: re-tokenized term and bounds of the main phrase in the term.
    """
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
    """ Inflect a source word using the PyMorphy2 library.

    :param source_parsed_word: the PyMorphy's parsed object for the source word.
    :param required_grammemes: set of required grammemes.
    :return: a string representation of inflected form and the PyMorphy's parsed object for this form.
    """
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
