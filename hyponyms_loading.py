import codecs
from typing import Dict, List, Tuple

from nltk import word_tokenize
import pymorphy2

from ruwordnet_parsing import noun_morphotag_to_str, verb_morphotag_to_str


def load_terms_for_submission(file_name: str) -> List[tuple]:
    line_idx = 1
    terms_list = []
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                new_term = tuple(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip().lower(), word_tokenize(prep_line))
                ))
                assert len(new_term) > 0, 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                terms_list.append(new_term)
            line_idx += 1
            cur_line = fp.readline()
    assert len(terms_list) > 0, 'File `{0}` is empty!'.format(file_name)
    return terms_list


def inflect_terms_for_submission(terms: List[tuple],
                                 main_pos_tag: str) -> Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]]:
    assert main_pos_tag in {"NOUN", "VERB"}
    morph = pymorphy2.MorphAnalyzer()
    grammemes = ([{"nomn"}, {"gent"}, {"datv"}, {"ablt"}, {"loct"}] if main_pos_tag == "NOUN"
                 else [{"past"}, {"pres"}, {"futr"}])
    res = dict()
    for id, cur in enumerate(terms):
        assert len(cur) == 1, 'The term `{0}` is wrong!'.format(' '.join(cur))
        term = cur[0]
        parsed = None
        for it in morph.parse(term):
            if (it.normal_form == term) and (str(it.tag.POS) == main_pos_tag):
                parsed = it
                break
        if (parsed is None) and (main_pos_tag == "NOUN"):
            for it in morph.parse(term):
                if (it.normal_form == term) and (str(it.tag.POS) == "ADJF"):
                    parsed = it
                    break
        if parsed is None:
            for it in morph.parse(term):
                if str(it.tag.POS) == main_pos_tag:
                    parsed = it
                    break
        assert parsed is not None, 'The term `{0}` cannot be parsed by the PyMorphy2!'.format(' '.join(cur))
        variants = dict()
        for grammeme in grammemes:
            morpho_data = parsed.inflect(grammeme)
            if morpho_data is not None:
                inflected = str(morpho_data.word)
                variants[(noun_morphotag_to_str(morpho_data) if main_pos_tag == "NOUN"
                          else verb_morphotag_to_str(morpho_data))] = ((inflected,), (0, 1))
        assert len(variants) > 0, 'The term `{0}` cannot be parsed by the PyMorphy2!'.format(' '.join(cur))
        res['{0}'.format(id)] = variants
    return res
