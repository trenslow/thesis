#!/usr/bin/python3
# This script takes as its input a file that lists the unique tokens found in a language's corpus and clusters them. The
# clustering is done by taking a pre-defined number of clusters from the most frequently occurring tokens in language
# 1 and assigning each token from language 2 to these clusters based on minimal Levenshtein distance or the Longest Common
# Substring. It then calculates the probabilities of a word given the class it belongs to.

import operator
from collections import Counter
from nltk.metrics import edit_distance


def read_train_file(lang, g2p):
    vocab = Counter()
    if g2p:
        with open('training_articles_g2p.' + lang, encoding='utf-8') as train:
            for line in train:
                ipa = ''.join(symb for symb in line.strip().split()[1:])
                vocab.update([ipa])
    else:
        with open('training_articles.' + lang, encoding='utf-8') as train:
            for w in train:
                vocab.update([w.strip()])

    return vocab


def write_vocab_file(lang, vocab, g2p):
    if g2p:
        voc = open('vocab_g2p.' + lang, 'w+')
    else:
        voc = open('vocab.' + lang, 'w+')
    for toke, cnt in sorted(vocab.items(), key=operator.itemgetter(1), reverse=True):
        voc.write(toke + ' ' + str(cnt) + '\n')
    voc.close()


def write_cluster_file(l1, l2, clusts, g2p):
    if g2p:
        clusters_out = open('clusters_g2p.' + l1 + '.' + l2, 'w+')
    else:
        clusters_out = open('clusters.' + l1 + '.' + l2, 'w+')

    for cluster, words in sorted(clusts.items(), key=operator.itemgetter(0)):
        clusters_out.write(cluster + ':\n')
        clusters_out.write(','.join(word + ' ' + str(count)
                           for word, count in sorted(words.items(), key=operator.itemgetter(1), reverse=True)))
        clusters_out.write('\n')

    clusters_out.close()


if __name__ == '__main__':
    lang1, lang2 = 'ces', 'pol'
    g2p = True
    lang1_vocab = read_train_file(lang1, g2p)
    most_freq_l1 = {t: c for t, c in sorted(lang1_vocab.items(), key=operator.itemgetter(1), reverse=True)[:100]}
    lang2_vocab = read_train_file(lang2, g2p)
    clusters = {c: {} for c in most_freq_l1}

    write_vocab_file(lang1, lang1_vocab, g2p)
    write_vocab_file(lang2, lang2_vocab, g2p)

    for word, count in lang2_vocab.items():
        old_dist = 1000
        old_clust = ''
        for c, cnt in sorted(most_freq_l1.items(), key=operator.itemgetter(1), reverse=True):
            new_dist = edit_distance(word, c)
            if new_dist < old_dist:
                old_dist = new_dist
                old_clust = c
        print(word, 'assigned to:', old_clust)
        clusters[old_clust][word] = count

    write_cluster_file(lang1, lang2, clusters, g2p)
