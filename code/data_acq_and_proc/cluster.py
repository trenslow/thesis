#!/usr/bin/python3
# This script takes as its input a file that lists the unique tokens found in a language's corpus and clusters them. The
# clustering is done by taking a pre-defined number of clusters from the most frequently occurring tokens in language
# 1 and assigning each token from language 2 to these clusters based on minimal Levenshtein distance or the Longest Common
# Substring. It then calculates the probabilities of a word given the class it belongs to.

import argparse
import operator
from collections import Counter
from nltk.metrics import edit_distance


def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def read_train_file(dir_path, lang, g2p):
    vocab = Counter()
    if g2p:
        with open(dir_path + 'training_articles_g2p.' + lang, encoding='utf-8') as train:
            for line in train:
                ipa = ''.join(symb for symb in line.strip().split()[1:])
                vocab.update([ipa])
    else:
        with open(dir_path + 'training_articles.' + lang, encoding='utf-8') as train:
            for w in train:
                vocab.update([w.strip()])

    return vocab


def write_vocab_file(dir_path, lang, vocab, g2p):
    if g2p:
        vocab_file = 'vocab_g2p.' + lang
    else:
        vocab_file = 'vocab.' + lang
    with open(dir_path + vocab_file, 'w+') as voc:
        for toke, cnt in sorted(vocab.items(), key=operator.itemgetter(1), reverse=True):
            voc.write(toke + ' ' + str(cnt) + '\n')


def write_cluster_file(dir_path, l1, l2, clusts, g2p, lcs):
    out_file_name = ''
    if g2p and lcs:
        out_file_name = '.'.join(['clusters_g2p_lcs', l1, l2])
    elif g2p:
        out_file_name = '.'.join(['clusters_g2p', l1, l2])
    elif lcs:
        out_file_name = '.'.join(['clusters_lcs', l1, l2])
    else:
        out_file_name = '.'.join(['clusters', l1, l2])

    with open(dir_path + out_file_name, 'w+') as clusters_out:
        for cluster, words in sorted(clusts.items(), key=operator.itemgetter(0)):
            clusters_out.write(cluster + ':\n')
            clusters_out.write(','.join(word + ' ' + str(count)
                               for word, count in sorted(words.items(), key=operator.itemgetter(1), reverse=True)))
            clusters_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1', help='the first language in a given pair')
    parser.add_argument('--lang2', help='the second language in a given pair')
    parser.add_argument('--g2p', default=False)
    parser.add_argument('--lcs', default=False)
    args = parser.parse_args()

    lang1, lang2 = args.lang1, args.lang2
    g2p = args.g2p
    lcs = args.lcs
    raw_data_dir = 'data/raw/'
    processed_data_dir = 'data/processed/'

    lang1_vocab = read_train_file(processed_data_dir, lang1, g2p)
    most_freq_l1 = {t: cnt for t, cnt in sorted(lang1_vocab.items(), key=operator.itemgetter(1), reverse=True)[:100]}
    lang2_vocab = read_train_file(processed_data_dir, lang2, g2p)
    clusters = {c: {} for c in most_freq_l1}

    write_vocab_file(processed_data_dir, lang1, lang1_vocab, g2p)
    write_vocab_file(processed_data_dir, lang2, lang2_vocab, g2p)

    sorted_clusters = sorted(most_freq_l1.items(), key=operator.itemgetter(1), reverse=True)
    for word, count in lang2_vocab.items():
        old_dist = 0 if lcs else 1000
        old_clust = ''
        for c, cnt in sorted_clusters:
            if lcs:
                new_dist = lcs_length(word, c)
                if new_dist > old_dist:
                    old_dist = new_dist
                    old_clust = c
            else:
                new_dist = edit_distance(word, c)
                if new_dist < old_dist:
                    old_dist = new_dist
                    old_clust = c
        if len(old_clust) == 0:  # assign word to most likely cluster to increase prob
            old_clust = sorted_clusters[0][0]
        # print(word, 'assigned to:', old_clust)
        clusters[old_clust][word] = count

    write_cluster_file(processed_data_dir, lang1, lang2, clusters, g2p, lcs)
