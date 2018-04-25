#!/usr/bin/python3
# This script takes as its input a corpus consisting of summaries of Wikipedia articles in two languages. It then takes
# these articles and processes the text into separate, lower-case tokens without punctuation. After doing so, the script
# takes the first 90% of articles as training data and the remaining 10% as test data.

import os
import string
import random


def split(articles):
    num_articles = len(articles)
    return articles[:int(0.9 * num_articles)], articles[num_articles - int(0.1 * num_articles):]


def write_train_test(lang, train, test):
    with open('data/training_articles.' + lang, 'w+') as train_out:
        for art in train:
            for word in art:
                train_out.write(word + '\n')
    with open('data/test_articles.' + lang, 'w+') as test_out:
        for art in test:
            for word in art:
                test_out.write(word + '\n')


if __name__ == '__main__':
    lang1, lang2 = 'bul', 'rus'
    corpus_path = 'wiki' + '_' + lang1 + '_' + lang2 + '/'
    exclude = set(string.punctuation)
    exclude.update({'„', '°', '“', '′', '«', '»', '×', '—', '–', '’', '‘', '”', 'ˈ', '″', 'ː'})
    lang1_articles = []
    lang2_articles = []

    for fn in sorted(os.listdir(corpus_path))[:]:
        with open('/nethome/trenslow/corpora/' + corpus_path + fn, encoding='utf-8') as article:
            print('collecting vocab from', fn)
            lines = []
            for line in article:
                tokens = line.strip().split()
                words = [''.join(char for char in token if char not in exclude) for token in tokens]
                clean_words = [word.lower() for word in words if word]
                lines += clean_words
            if lang1 in fn:
                lang1_articles.append(lines)
            else:
                lang2_articles.append(lines)

    print('splitting articles for ' + lang1)
    lang1_train, lang1_test = split(lang1_articles)
    print('splitting articles for ' + lang2)
    lang2_train, lang2_test = split(lang2_articles)
    write_train_test(lang1, lang1_train, lang1_test)
    write_train_test(lang2, lang2_train, lang2_test)
