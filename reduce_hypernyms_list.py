from argparse import ArgumentParser
import codecs
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_hypernyms', type=str, required=True,
                        help='A file with full resulted data, i.e. a lot of found hypernyms for '
                             'each corresponded hyponym.')
    parser.add_argument('-d', '--dst', dest='reduced_hypernyms', type=str, required=True,
                        help='A file with reduced resulted data, i.e. ten hypernyms for each corresponded hyponym.')
    args = parser.parse_args()

    source_file_name = os.path.normpath(args.source_hypernyms)
    destination_file_name = os.path.normpath(args.reduced_hypernyms)
    assert os.path.isfile(source_file_name), 'A file `{0}` does not exist!'.format(source_file_name)
    destination_file_dir = os.path.dirname(destination_file_name)
    if len(destination_file_dir) > 0:
        assert os.path.isdir(destination_file_dir), 'A directory `{0}` does not exist!'.format(destination_file_dir)

    source_hypernyms_by_hyponyms = dict()
    line_idx = 1
    with codecs.open(source_file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(source_file_name, line_idx)
                line_parts = prep_line.split('\t')
                assert len(line_parts) in {2, 3}, err_msg
                hyponym_value = line_parts[0]
                hypernym_id = line_parts[1]
                hypernym_value = '' if len(line_parts) == 2 else line_parts[2]
                if hyponym_value in source_hypernyms_by_hyponyms:
                    if len(source_hypernyms_by_hyponyms[hyponym_value]) < 10:
                        source_hypernyms_by_hyponyms[hyponym_value].append((hypernym_id, hypernym_value))
                else:
                    source_hypernyms_by_hyponyms[hyponym_value] = [(hypernym_id, hypernym_value)]
            cur_line = fp.readline()
            line_idx += 1
    with codecs.open(destination_file_name, mode="w", encoding="utf-8", errors="ignore") as fp:
        for hyponym_value in sorted(list(source_hypernyms_by_hyponyms.keys())):
            for hypernym_id, hypernym_value in source_hypernyms_by_hyponyms[hyponym_value]:
                if len(hypernym_value) > 0:
                    fp.write('{0}\t{1}\t{2}\n'.format(hyponym_value, hypernym_id, hypernym_value))
                else:
                    fp.write('{0}\t{1}\n'.format(hyponym_value, hypernym_id))


if __name__ == '__main__':
    main()
