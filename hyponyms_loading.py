import codecs
from typing import List

from nltk import word_tokenize


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
