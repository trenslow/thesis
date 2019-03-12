from itertools import islice
import re
from collections import OrderedDict as OD
from nltk.metrics import edit_distance
import sys


def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def get_clusters(l1, l2, g2p):
    if not g2p:
        with open('data/swadesh.' + l1, encoding='utf-8') as cluster_file:
            return {line.strip() for line in cluster_file}
    else:
        with open('data/swadesh_g2p.' + l1, encoding='utf-8') as cluster_file:
            return {''.join(char for char in line.strip().split()[1:]) for line in cluster_file}


def make_g2p_map(l2):
    g2p = OD()
    old_graph_len = 0
    g2p_in = 'training_articles_g2p.' + l2
    g2p_err_in = 'training_articles_g2p.' + l2 + '.err'

    with open('data/'+ g2p_in, encoding='utf-8') as l2s:
        for line in l2s.readlines():
            split = line.strip().split()
            grapheme_rep, phoneme_rep = split[0], ''.join(char for char in split[1:])
            new_graph_len = len(grapheme_rep)
            if new_graph_len > old_graph_len:
                old_graph_len = len(grapheme_rep)
            if len(phoneme_rep) > 0:
                g2p[grapheme_rep] = phoneme_rep
            else:
                g2p[grapheme_rep] = None
                
    with open('data/' + g2p_err_in, encoding='utf-8') as l2f:
        for line in l2f.readlines()[:-1]:
            fail = re.findall(r'"(.*?)"', line)[0]
            fail_len = len(fail)
            if fail_len > old_graph_len:
                old_graph_len = fail_len
            g2p[fail] = None

    return g2p, old_graph_len + 1


def minimize_edit_distance(clusters, rep):
    old_dist = 1000
    for c in clusters:
        new_dist = edit_distance(rep, c)
        if new_dist < old_dist:
            old_dist = new_dist
    return old_dist


def maximize_common_substring(clusters, rep):
    old_dist = 0
    for c in clusters:
        new_dist = lcs_length(rep, c)
        if new_dist > old_dist:
            old_dist = new_dist
    return old_dist


if __name__ == '__main__':
    # code changed for swadesh clusters
    lang1, lang2 = sys.argv[1:3]
    l1_clusters = get_clusters(lang1, lang2, False)
    l1_g2p_clusters = get_clusters(lang1, lang2, True)
    g2p_map, ed_ceiling = make_g2p_map(lang2)
    data_dir = 'data/'
    in_file = data_dir + 'training_articles.' + lang2
    out_file = data_dir + 'vectors_swadesh.' + lang1 + '.' + lang2

    with open(in_file, encoding='utf-8') as inf, open(out_file, 'w+', encoding='utf-8') as out:
        for line in inf:
            graph_rep = line.strip()
            min_ed_graph = minimize_edit_distance(l1_clusters, graph_rep)
            max_cs_graph = maximize_common_substring(l1_clusters, graph_rep)
            phone_rep = g2p_map[graph_rep]
            if phone_rep:
                min_ed_phone = minimize_edit_distance(l1_g2p_clusters, phone_rep)
                max_cs_phone = maximize_common_substring(l1_g2p_clusters, phone_rep)
            else:
                min_ed_phone = ed_ceiling
                max_cs_phone = 0

            vector = [min_ed_graph, max_cs_graph, min_ed_phone, max_cs_phone, graph_rep]
            out.write(' '.join(str(el) for el in vector) + '\n')