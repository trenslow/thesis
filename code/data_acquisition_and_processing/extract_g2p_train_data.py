import operator
import string
import random
import os


def read_in(file):
    bul, ces, pol, rus = {}, {}, {}, {}
    with open(file) as f:
        for line in f:
            split = line.strip().split('\t')
            word = split[2].lower()
            if ':' in word:
                continue
            else:
                clean_word = word.replace(' ', '#')
            lang = split[0]
            ipa = split[3]

            if lang == 'bul':
                if clean_word not in bul:
                    bul[clean_word] = ipa
                else:
                    continue
            elif lang == 'ces':
                if clean_word not in ces:
                    ces[clean_word] = ipa
                else:
                    continue
            elif lang == 'pol':
                if clean_word not in pol:
                    pol[clean_word] = ipa
                else:
                    continue
            elif lang == 'rus':
                if clean_word not in rus:
                    rus[clean_word] = ipa
                else:
                    continue

    return bul, ces, pol, rus


def write_g2p(out_dir, data, lang):
    sorted_data = sorted(data.items(), key=operator.itemgetter(0))
    random.shuffle(sorted_data)
    num_words = len(data)
    train, test = sorted_data[:int(0.95 * num_words)], sorted_data[num_words - int(0.05 * num_words):]

    with open(out_dir + '.'.join(['training_articles_ipa', lang])  , 'w+') as tr:
        for t in train:
            tr.write(t[0] + ' ' + t[1] + '\n')

    with open(out_dir + '.'.join(['test_articles_ipa', lang]), 'w+') as te:
        for t in test:
            te.write(t[0] + ' ' + t[1] + '\n')


if __name__ == '__main__':
    processed_data_dir = 'data/processed/'
    all_file_path = 'data/raw/pron_data/all.phoible'
    bul_data, ces_data, pol_data, rus_data = read_in(all_file_path)

    write_g2p(processed_data_dir, bul_data, 'bul')
    write_g2p(processed_data_dir, ces_data, 'ces')
    write_g2p(processed_data_dir, pol_data, 'pol')
    write_g2p(processed_data_dir, rus_data, 'rus')