"""
This module is a part of system for the automatic enrichment
of a WordNet-like taxonomy.

Code of this module is based on Andrey Kutuzov's tutorial
https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb

Copyright 2020 Ivan Bondarenko

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

import os
from typing import List, Tuple, Union

from ufal.udpipe import Model, Pipeline, ProcessingError


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


def initialize_udpipe(model_name: str) -> Tuple[Pipeline, ProcessingError]:
    udpipe_model_name = os.path.normpath(model_name)
    assert os.path.isfile(udpipe_model_name), 'A file "{0}" does not exist!'.format(udpipe_model_name)
    udpipe_model = Model.load(udpipe_model_name)
    udpipe_pipeline = Pipeline(udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    udpipe_error = ProcessingError()
    print('The UDPipe pipeline has been prepared...')
    print('')
    return udpipe_pipeline, udpipe_error
