from argparse import ArgumentParser
import codecs
import os
import random
import re
from typing import Dict, List, Set, Tuple, Union

from lxml import etree
import nltk
import numpy as np
from ufal.udpipe import Model, Pipeline, ProcessingError

from text_processing import load_news, load_wiki, tokenize


def load_senses_from_ruwordnet(file_name: str) -> Tuple[List[str], Dict[str, Set[int]]]:
    with open(file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    lemmatized_terms = []
    for sense in root.getchildren():
        if sense.tag == 'sense':
            sense_id = sense.get('id').strip()
            assert len(sense_id) > 0
            synset_id = sense.get('synset_id').strip()
            assert len(synset_id) > 0
            assert sense_id.startswith(synset_id)
            term = sense.get('name').strip()
            assert len(term) > 0
            term = tuple(filter(lambda it2: (len(it2) > 0) and it2.isalnum(), map(lambda it1: it1.strip().lower(),
                                                                                  tokenize(term))))
            assert len(term) > 0
            lemma = sense.get('lemma').strip()
            assert len(lemma) > 0
            lemma = tuple(filter(lambda it2: (len(it2) > 0) and it2.isalnum(), map(lambda it1: it1.strip().lower(),
                                                                                   tokenize(lemma))))
            assert len(lemma) > 0
            lemmatized_terms.append(lemma)
    del xml_data, root
    assert len(lemmatized_terms) > 0
    words = dict()
    for term_idx, cur_term in enumerate(lemmatized_terms):
        for cur_word in cur_term:
            if cur_word in words:
                words[cur_word].add(term_idx)
            else:
                words[cur_word] = {term_idx}
    return list(map(lambda it: ' '.join(it), lemmatized_terms)), words


def load_unseen_hyponyms(file_name: str, udpipe_pipeline: Pipeline,
                         udpipe_error: ProcessingError) -> Tuple[List[str], Dict[str, Set[int]]]:
    lemmatized_terms = []
    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                source_term = tokenize(prep_line, lowercased=True)
                prepared_term = tuple(filter(
                    lambda it2: (len(it2) > 0) and it2.isalnum(),
                    map(lambda it1: it1.strip().lower(), source_term)
                ))
                assert len(prepared_term) > 0
                if len(prepared_term) > 1:
                    normalized_term = process_with_udpipe(pipeline=udpipe_pipeline, error=udpipe_error,
                                                          text=' '.join(source_term), keep_pos=False, keep_punct=False)
                    normalized_term = tuple(filter(
                        lambda it2: (len(it2) > 0) and it2.isalnum(),
                        map(lambda it1: it1.strip().lower(), normalized_term)
                    ))
                    assert len(normalized_term) > 0
                    lemmatized_terms.append(normalized_term)
                else:
                    lemmatized_terms.append(prepared_term)
            cur_line = fp.readline()
    assert len(lemmatized_terms) > 0
    words = dict()
    for term_idx, cur_term in enumerate(lemmatized_terms):
        for cur_word in cur_term:
            if cur_word in words:
                words[cur_word].add(term_idx)
            else:
                words[cur_word] = {term_idx}
    return list(map(lambda it: ' '.join(it), lemmatized_terms)), words


def num_replace(word: str) -> str:
    newtoken = 'x' * len(word)
    return newtoken


def clean_token(token: str, misc: str) -> Union[str, None]:
    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def clean_lemma(lemma: str, pos: str) -> Union[str, None]:
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])
        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])
        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])
    return out_lemma


def process_with_udpipe(pipeline: Pipeline, error: ProcessingError, text: str, keep_pos: bool=True,
                        keep_punct: bool=False) -> List[str]:
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []
    processed = pipeline.process(text, error)
    assert not error.occurred(), 'The text `{0}` cannot be processed with UDPipe!'.format(text)
    content = [l for l in processed.split('\n') if not l.startswith('#')]
    tagged = [w.split('\t') for w in content if w]
    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        token = clean_token(token, misc)
        lemma = clean_lemma(lemma, pos)
        if not lemma or not token:
            continue
        if pos in entities:
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))
        else:
            if not named:
                if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                    lemma = num_replace(token)
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))

    if not keep_punct:
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos:
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    return tagged_propn


def remove_pos_tags(lemmatized_tokens: List[str]) -> str:
    re_for_pos_tag = re.compile(r'_[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]+$')
    tokens = list(filter(
        lambda it2: len(it2) > 0,
        map(lambda it1: re_for_pos_tag.sub('', it1).strip().lower(), lemmatized_tokens)
    ))
    return ' ' + ' '.join(tokens) + ' '


def main():
    random.seed(142)
    np.random.seed(142)

    parser = ArgumentParser()
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-i', '--input', dest='input_dir', type=str, required=True,
                        help='A directory with input data as a list of unseen hyponyms for public and private sets.')
    parser.add_argument('-r', '--res', dest='resulted_text_corpus', type=str, required=True,
                        help='Resulted text file into which a text corpus will be written.')
    parser.add_argument('-d', '--data', dest='data_source', type=str, required=True,
                        choices=['wiki', 'news', 'librusec'],
                        help='A data source kind (wiki, news or librusec), prepared for '
                             'the Taxonomy Enrichment competition.')
    parser.add_argument('-p', '--path', dest='source_path', required=True, type=str,
                        help='Path to the source data file or directory.')
    parser.add_argument('-u', '--udpipe', dest='udpipe_model', required=True, type=str,
                        help='Path to the UDPipe model.')
    args = parser.parse_args()

    nltk.download('punkt')
    wordnet_dir = os.path.normpath(args.wordnet_dir)
    assert os.path.isdir(wordnet_dir), 'A directory `{0}` does not exist!'.format(wordnet_dir)
    wordnet_senses_name = os.path.join(wordnet_dir, 'senses.N.xml' if args.track_name == 'nouns' else 'senses.V.xml')
    wordnet_synsets_name = os.path.join(wordnet_dir, 'synsets.N.xml' if args.track_name == 'nouns' else 'synsets.V.xml')
    wordnet_relations_name = os.path.join(
        wordnet_dir,
        'synset_relations.N.xml' if args.track_name == 'nouns' else 'synset_relations.V.xml'
    )
    assert os.path.isfile(wordnet_senses_name), 'A file `{0}` does not exist!'.format(wordnet_senses_name)
    assert os.path.isfile(wordnet_synsets_name), 'A file `{0}` does not exist!'.format(wordnet_synsets_name)
    assert os.path.isfile(wordnet_relations_name), 'A file `{0}` does not exist!'.format(wordnet_relations_name)

    input_dir = os.path.normpath(args.input_dir)
    assert os.path.isdir(input_dir), 'A directory `{0}` does not exist!'.format(input_dir)
    public_data_name = os.path.join(input_dir,
                                    '{0}_public.tsv'.format('nouns' if args.track_name == 'nouns' else 'verbs'))
    assert os.path.isfile(public_data_name), 'A file `{0}` does not exist!'.format(public_data_name)
    private_data_name = os.path.join(input_dir,
                                     '{0}_private.tsv'.format('nouns' if args.track_name == 'nouns' else 'verbs'))
    assert os.path.isfile(private_data_name), 'A file `{0}` does not exist!'.format(private_data_name)

    text_corpus_name = os.path.normpath(args.resulted_text_corpus)
    text_corpus_dir = os.path.dirname(text_corpus_name)
    assert os.path.isdir(text_corpus_dir), 'A directory `{0}` does not exist!'.format(text_corpus_dir)

    assert args.data_source != "librusec", "The processing of the LibRuSec text corpus is not implemented already!"
    full_path = os.path.normpath(args.source_path)
    if args.data_source == "news":
        assert os.path.isdir(full_path), 'A directory "{0}" does not exist!'.format(full_path)
    else:
        assert os.path.isfile(full_path), 'A file "{0}" does not exist!'.format(full_path)

    udpipe_model_name = os.path.normpath(args.udpipe_model)
    assert os.path.isfile(udpipe_model_name), 'A file "{0}" does not exist!'.format(udpipe_model_name)

    udpipe_model = Model.load(udpipe_model_name)
    udpipe_pipeline = Pipeline(udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    udpipe_error = ProcessingError()
    print('The UDPipe pipeline has been prepared...')
    print('')

    terms_from_ruwordnet, ruwordnet_search_index = load_senses_from_ruwordnet(wordnet_senses_name)
    print('{0} terms have been loaded from the RuWordNet.'.format(len(terms_from_ruwordnet)))
    terms_from_public, public_search_index = load_unseen_hyponyms(public_data_name, udpipe_pipeline=udpipe_pipeline,
                                                                  udpipe_error=udpipe_error)
    print('{0} terms have been loaded from the public set.'.format(len(terms_from_public)))
    terms_from_private, private_search_index = load_unseen_hyponyms(private_data_name, udpipe_pipeline=udpipe_pipeline,
                                                                    udpipe_error=udpipe_error)
    print('{0} terms have been loaded from the private set.'.format(len(terms_from_private)))
    print('')

    generator = load_news(full_path) if args.data_source == "news" else load_wiki(full_path)
    source_texts_counter = 0
    saved_texts_counter = 0
    frequencies_of_ruwordnet_terms = dict()
    frequencies_of_private_terms = dict()
    frequencies_of_public_terms = dict()
    with codecs.open(text_corpus_name, mode="a", encoding="utf-8", errors="ignore") as fp:
        for new_text in generator:
            source_tokens = tokenize(new_text, lowercased=False)
            prepared_tokens = tuple(filter(
                lambda it2: (len(it2) > 0) and it2.isalnum(),
                map(lambda it1: it1.strip().lower(), source_tokens)
            ))
            if len(prepared_tokens) >= 10:
                lemmatized_tokens = process_with_udpipe(pipeline=udpipe_pipeline, error=udpipe_error,
                                                        text=' '.join(source_tokens), keep_pos=False, keep_punct=False)
                lemmatized_tokens = tuple(filter(
                    lambda it2: (len(it2) > 0) and it2.isalnum(),
                    map(lambda it1: it1.strip().lower(), lemmatized_tokens)
                ))
                lemmatized_text = ' ' + ' '.join(lemmatized_tokens) + ' '
                indices_of_ruwordnet_terms = set()
                indices_of_public_terms = set()
                indices_of_private_terms = set()
                for cur_token in lemmatized_tokens:
                    if cur_token in ruwordnet_search_index:
                        indices_of_ruwordnet_terms |= ruwordnet_search_index[cur_token]
                    if cur_token in public_search_index:
                        indices_of_public_terms |= public_search_index[cur_token]
                    if cur_token in private_search_index:
                        indices_of_private_terms |= private_search_index[cur_token]
                ok = False
                bounds_of_terms = []
                used_characters = [0 for _ in range(len(lemmatized_text))]
                if len(indices_of_public_terms) > 0:
                    for term_idx in indices_of_public_terms:
                        term = terms_from_public[term_idx]
                        term_with_spaces = ' ' + term + ' '
                        found_pos = lemmatized_text.find(term_with_spaces)
                        if found_pos >= 0:
                            bounds_of_new_term = (found_pos + 1, found_pos + len(term_with_spaces) - 1)
                            if all(map(lambda char_idx: used_characters[char_idx] == 0,
                                       range(bounds_of_new_term[0], bounds_of_new_term[1]))):
                                ok = True
                                frequencies_of_public_terms[term] = frequencies_of_public_terms.get(term, 0) + 1
                                bounds_of_terms.append(bounds_of_new_term)
                                for char_idx in range(bounds_of_new_term[0], bounds_of_new_term[1]):
                                    used_characters[char_idx] = 1
                if len(indices_of_private_terms) > 0:
                    for term_idx in indices_of_private_terms:
                        term = terms_from_private[term_idx]
                        term_with_spaces = ' ' + term + ' '
                        found_pos = lemmatized_text.find(term_with_spaces)
                        if found_pos >= 0:
                            bounds_of_new_term = (found_pos + 1, found_pos + len(term_with_spaces) - 1)
                            if all(map(lambda char_idx: used_characters[char_idx] == 0,
                                       range(bounds_of_new_term[0], bounds_of_new_term[1]))):
                                ok = True
                                frequencies_of_private_terms[term] = frequencies_of_private_terms.get(term, 0) + 1
                                bounds_of_terms.append(bounds_of_new_term)
                                for char_idx in range(bounds_of_new_term[0], bounds_of_new_term[1]):
                                    used_characters[char_idx] = 1
                if len(indices_of_ruwordnet_terms) > 0:
                    for term_idx in indices_of_ruwordnet_terms:
                        term = terms_from_ruwordnet[term_idx]
                        term_with_spaces = ' ' + term + ' '
                        found_pos = lemmatized_text.find(term_with_spaces)
                        if found_pos >= 0:
                            bounds_of_new_term = (found_pos + 1, found_pos + len(term_with_spaces) - 1)
                            if all(map(lambda char_idx: used_characters[char_idx] == 0,
                                       range(bounds_of_new_term[0], bounds_of_new_term[1]))):
                                ok = True
                                frequencies_of_ruwordnet_terms[term] = frequencies_of_ruwordnet_terms.get(term, 0) + 1
                                bounds_of_terms.append(bounds_of_new_term)
                                for char_idx in range(bounds_of_new_term[0], bounds_of_new_term[1]):
                                    used_characters[char_idx] = 1
                if ok:
                    assert len(bounds_of_terms) > 0
                    for term_start, term_end in bounds_of_terms:
                        lemmatized_text = lemmatized_text[:term_start] + \
                                          lemmatized_text[term_start:term_end].replace(' ', '_') + \
                                          lemmatized_text[term_end:]
                    fp.write('{0}\n'.format(lemmatized_text.strip().replace('ё', 'е')))
                    saved_texts_counter += 1
                source_texts_counter += 1
                del indices_of_ruwordnet_terms, indices_of_private_terms, indices_of_public_terms
                del lemmatized_tokens, used_characters, bounds_of_terms
                del lemmatized_text, new_text
            del source_tokens, prepared_tokens
            if (source_texts_counter % 10000 == 0) and (source_texts_counter > 0):
                print('{0} texts have been processed...'.format(source_texts_counter))
                print('{0} texts have been saved into the text corpus...'.format(saved_texts_counter))
                unknown_terms_from_ruwordnet = set(terms_from_ruwordnet) - set(frequencies_of_ruwordnet_terms.keys())
                print('{0} terms from the RuWordNet are unknown.'.format(len(unknown_terms_from_ruwordnet)))
                unknown_public_terms = set(terms_from_public) - set(frequencies_of_public_terms.keys())
                print('{0} public terms are unknown.'.format(len(unknown_public_terms)))
                unknown_private_terms = set(terms_from_private) - set(frequencies_of_private_terms.keys())
                print('{0} private terms are unknown.'.format(len(unknown_private_terms)))
                print('')
    if source_texts_counter % 10000 != 0:
        print('{0} texts have been processed...'.format(source_texts_counter))
        print('{0} texts have been saved into the text corpus...'.format(saved_texts_counter))
        unknown_terms_from_ruwordnet = set(terms_from_ruwordnet) - set(frequencies_of_ruwordnet_terms.keys())
        print('{0} terms from the RuWordNet are unknown.'.format(len(unknown_terms_from_ruwordnet)))
        unknown_public_terms = set(terms_from_public) - set(frequencies_of_public_terms.keys())
        print('{0} public terms are unknown.'.format(len(unknown_public_terms)))
        unknown_private_terms = set(terms_from_private) - set(frequencies_of_private_terms.keys())
        print('{0} private terms are unknown.'.format(len(unknown_private_terms)))


if __name__ == '__main__':
    main()
