#!/usr/bin/python3
# This script takes as its input a text file that lists the titles of all Wikipedia articles that are shared by two
# languages. It then uses these article titles to download and write the summaries from each article into separate files
# that form a corpus.
# In cases where the article title produces Wikipedia's disambiguation page, where the article has no summary, where the
# page no longer exists, or where the Wikipedia package fails to decode the summary, these articles are simply skipped.

import wikipedia
import os
import sys
import argparse
import operator
import json


def read_article_titles(file):
    title_pairs = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            ln = line.strip().split(' ||| ')
            title_pairs.append((ln[0], ln[1]))
    return title_pairs


def scrape(idx, lang1, lang2, corp_path):
    wikipedia.set_rate_limiting(True)
    lang1_disamb_count, lang2_disamb_count = 0, 0
    lang1_no_summ_count, lang2_no_summ_count = 0, 0
    for i, pair in sorted(idx.items(), key=operator.itemgetter(0)):
        if i % 10 == 0:
            print("last update at article:", i)
        wikipedia.set_lang(lang1)
        try:
            lang1_summ = wikipedia.summary(pair[0])
        except wikipedia.DisambiguationError:
            lang1_disamb_count += 1
            print("skipping " + lang1 + " article: " + pair[0])
            print("number of disambiguations in", lang1, "until now:", lang1_disamb_count)
            continue
        except wikipedia.PageError:
            print(pair[0], "doesn't exist in", lang1, "!")
            continue
        except json.decoder.JSONDecodeError:
            print("json decode error")
            continue
        if lang1_summ:
            with open(corp_path + '.'.join([str(i), lang1]), 'w+') as out1:
                out1.write(lang1_summ)
            continue
        else:
            lang1_no_summ_count += 1
            print(pair[0], "doesn't have a summary in", lang1, "!")
            print("number of empty summaries in", lang1, "until now:", lang1_no_summ_count)

        wikipedia.set_lang(lang2)
        try:
            lang2_summ = wikipedia.summary(pair[1])
        except wikipedia.DisambiguationError:
            lang2_disamb_count += 1
            print("skipping " + lang2 + " article: " + pair[1])
            print("number of disambiguations in", lang2, "until now:", lang2_disamb_count)
            continue
        except wikipedia.PageError:
            print(pair[1], "doesn't exist in", lang2, "!")
            continue
        except json.decoder.JSONDecodeError:
            print("json decode error")
            continue
        if lang2_summ:
            with open(corp_path + '.'.join([str(i), lang2]), 'w+') as out2:
                out2.write(lang2_summ)
            continue
        else:
            lang2_no_summ_count += 1
            print(pair[1], "doesn't have a summary in", lang2, "!")
            print("number of empty summaries in", lang2, "until now:", lang2_no_summ_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1', help='the first language in a given pair')
    parser.add_argument('--lang2', help='the second language in a given pair')
    parser.add_argument('--corpora-path', help='location of corpora directory on disk')
    args = parser.parse_args()
    lang1, lang2 = args.lang1, args.lang2
    corpora_path = args.corpora_path
    if not os.path.exists(corpora_path):
        os.makedirs(corpora_path)

    titles = read_article_titles(corpora_path + '.'.join(['titles', lang1, lang2]) )
    index = {i: pair for i, pair in enumerate(titles)}

    iso_map = {
        'bul': 'bg',
        'ces': 'cs',
        'pol': 'pl',
        'rus': 'ru'
    }

    corpus_dir_name = corpora_path + 'wiki' + '_' + lang1 + '_' + lang2 + '/'
    scrape(index, iso_map[lang1], iso_map[lang2], corpus_dir_name)

