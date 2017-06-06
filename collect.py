#!/usr/bin/python3
# This script takes as its input a corpus consisting of summaries of Wikipedia articles in two languages. It then takes
# these articles and processes the text into separate, lower-case tokens without punctuation. After doing so, the script
# breaks up the articles into training and test data. It can do this split based on finding an OOV rate
# between training and test data of less than 5%, or it can simply split the articles into a pre-defined percentage of
# training and test data. After the split, it writes the training and test data of each language to separate files.

import os
import string
import random


def optimal_split(articles):
    num_articles = len(articles)
    inc = 0.05
    train_portion = 0.5
    test_portion = 0.5
    train = articles[:int(train_portion * num_articles)]
    test = articles[num_articles - int(test_portion * num_articles):]

    while train_portion <= 0.95:
        train_words = [word for art in train for word in art]
        test_words = [word for art in test for word in art]
        length_train = len(train_words)
        num_oov = len([t for t in test_words if t not in train_words])
        # print(length_train, num_oov)
        new_oov_rate = num_oov / length_train
        if new_oov_rate > 0.05:
            oov_rate = new_oov_rate
            print('new intralanguage oov rate: {:.2%}'.format(oov_rate))
            train_portion += inc
            test_portion -= inc
            train = articles[:int(train_portion * num_articles)]
            test = articles[num_articles - int(test_portion * num_articles):]
        else:
            print('oov now under 5%: {:.2%}'.format(new_oov_rate))
            print('train and test split at {:.0%} and {:.0%} respectively'.format(train_portion, test_portion))
            break

    return train, test


def split(articles):
    num_articles = len(articles)
    return articles[:int(0.9 * num_articles)], articles[num_articles - int(0.1 * num_articles):]


def write_train_test(lang, train, test):
    with open('training_articles.' + lang, 'w+') as train_out:
        for art in train:
            for word in art:
                train_out.write(word + '\n')
    with open('test_articles.' + lang, 'w+') as test_out:
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
        with open('/home/tyler/PycharmProjects/thesis/' + corpus_path + fn, encoding='utf-8') as article:
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

    random.shuffle(lang1_articles)
    random.shuffle(lang2_articles)
    # lang1_train, lang1_test = optimal_split(lang1_articles)
    # lang2_train, lang2_test = optimal_split(lang2_articles)
    lang1_train, lang1_test = split(lang1_articles)
    lang2_train, lang2_test = split(lang2_articles)
    write_train_test(lang1, lang1_train, lang1_test)
    write_train_test(lang2, lang2_train, lang2_test)
