import operator
import random


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
            original_ipa = split[4]

            if lang == 'bul':
                if clean_word not in bul:
                    bul[clean_word] = []
                    bul[clean_word].append(ipa)
                else:
                    bul[clean_word].append(ipa)
                    continue
            elif lang == 'ces':
                if clean_word not in ces:
                    ces[clean_word] = []
                    ces[clean_word].append(ipa)
                else:
                    ces[clean_word].append(ipa)
                    continue
            elif lang == 'pol':
                if clean_word not in pol:
                    pol[clean_word] = []
                    pol[clean_word].append(ipa)
                else:
                    pol[clean_word].append(ipa)
                    continue
            elif lang == 'rus':
                if clean_word not in rus:
                    rus[clean_word] = []
                    rus[clean_word].append(ipa)
                else:
                    rus[clean_word].append(ipa)
                    continue

    return bul, ces, pol, rus


def write_g2p(data, language):
    sorted_data = sorted(data.items(), key=operator.itemgetter(0))
    random.shuffle(sorted_data)
    num_words = len(data)
    train, test = sorted_data[:int(0.95 * num_words)], sorted_data[num_words - int(0.05 * num_words):]

    with open('g2p_training.' + language, 'w+') as tr:
        for t in train:
            tr.write(t[0] + ' ' + t[1] + '\n')

    with open('g2p_test.' + language, 'w+') as te:
        for t in test:
            te.write(t[0] + ' ' + t[1] + '\n')


if __name__ == '__main__':
    all_file = 'all.phoible'
    bulgarian, czech, polish, russian = read_in(all_file)

    write_g2p(bulgarian, 'bul')
    write_g2p(czech, 'ces')
    write_g2p(polish, 'pol')
    write_g2p(russian, 'rus')