import gzip
import json
import os
import re


def load_news(corpus_dir_name: str):
    re_for_filename = re.compile(r'^news_df_\d+.csv$')
    data_files = list(map(
        lambda it2: os.path.join(os.path.normpath(corpus_dir_name), it2),
        filter(
            lambda it1: re_for_filename.match(it1.lower()) is not None,
            os.listdir(os.path.normpath(corpus_dir_name))
        )
    ))
    assert len(data_files) > 0
    for cur_file in data_files:
        with gzip.open(cur_file, mode="rt") as fp:
            cur_line = fp.readline()
            line_idx = 1
            true_header = ["file_name", "file_sentences"]
            loaded_header = []
            while len(cur_line) > 0:
                prep_line = cur_file.strip()
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
                            tokens = tuple(filter(
                                lambda it2: len(it2) > 0,
                                map(lambda it1: it1.strip(), text.split())
                            ))
                            assert len(tokens) > 0, err_msg
                            yield tokens
                cur_line = fp.readline()
                line_idx += 1

