from itertools import islice
import re
import argparse
from collections import OrderedDict as OD
from nltk.metrics import edit_distance
import os


def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


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


def get_clusters(cluster_dir, base_name, l1, l2, clusts):
    if clusts == 0:
        path_to_cluster_file = cluster_dir + '.'.join([base_name, l1, l2])
        with open(path_to_cluster_file, encoding='utf-8') as clusters:
            return {cluster.strip()[:-1] for cluster in islice(clusters, 0, None, 2)}
    elif clusts == 1:
        path_to_cluster_file = cluster_dir + '.'.join([base_name + '_g2p', l1, l2])
        with open(path_to_cluster_file, encoding='utf-8') as ipa_clusters:
            return {cluster.strip()[:-1] for cluster in islice(ipa_clusters, 0, None, 2)}
    elif clusts == 2:
        path_to_cluster_file = cluster_dir + '.'.join([base_name + '_reconstructed', l1, l2])
        with open(path_to_cluster_file, encoding='utf-8') as recon_clusters:
            recons = set()
            for line in recon_clusters:
                split = line.strip().split()
                if len(split) == 2:  # if successful xformation, take xformed version
                    recons.add(split[1])
                else:  # if unsuccessful xformation, take original word
                    recons.add(split[0])
            return recons


def make_g2p_map(g2p_dir, l2, tst):
    g2p_map = {}
    old_graph_len = 0
    if tst:
        g2p_path = 'g2p_test_data.' + l2
        g2p_err_path = g2p_path + '.err'
    else:
        g2p_path = 'g2p_train_data.' + l2
        g2p_err_path = g2p_path + '.err'

    with open(g2p_dir + g2p_path, encoding='utf-8') as l2s:
        for line in l2s.readlines():
            split = line.strip().split()
            grapheme_rep, phoneme_rep = split[0], ''.join(char for char in split[1:])
            new_graph_len = len(grapheme_rep)
            if new_graph_len > old_graph_len:
                old_graph_len = len(grapheme_rep)
            if len(phoneme_rep) > 0:
                g2p_map[grapheme_rep] = phoneme_rep
            else:
                g2p_map[grapheme_rep] = None
                
    with open(g2p_dir + g2p_err_path, encoding='utf-8') as l2f:
        for line in l2f.readlines()[:-1]:
            fail = re.findall(r'"(.*?)"', line)[0]
            fail_len = len(fail)
            if fail_len > old_graph_len:
                old_graph_len = fail_len
            g2p_map[fail] = None

    return g2p_map, old_graph_len + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1', help='the first language in a given pair')
    parser.add_argument('--lang2', help='the second language in a given pair')
    parser.add_argument('--swadesh', help='whether to create vectors based on swadesh clusters', default=False)
    args = parser.parse_args()

    lang1, lang2 = args.lang1, args.lang2
    swadesh = args.swadesh
    proj_root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    processed_data_dir = os.path.join(proj_root_dir, 'data', 'processed/')
    modeling_data_dir = os.path.join(proj_root_dir, 'data', 'modeling/')
    train_data_path = processed_data_dir + '.'.join(['training_articles', lang2])
    test_data_path = processed_data_dir + '.'.join(['test_articles', lang2])
    if swadesh:
        train_feat_vec_path = modeling_data_dir + '.'.join(['train_vectors_swadesh', lang1, lang2])
        test_feat_vec_path = modeling_data_dir + '.'.join(['test_vectors_swadesh', lang1, lang2])
    else:
        train_feat_vec_path = modeling_data_dir + '.'.join(['train_vectors', lang1, lang2])
        test_feat_vec_path = modeling_data_dir + '.'.join(['test_vectors', lang1, lang2])

    all_cluster_sets = OD(
        [
            ('word', {}),
            ('ipa', {}),
            ('reconstructed', {})
        ]
    )
    if swadesh:
        file_base_name = 'swadesh'
        all_cluster_sets['word'] = get_clusters(processed_data_dir, file_base_name, lang1, lang2, 0)
        all_cluster_sets['ipa'] = get_clusters(processed_data_dir, file_base_name, lang1, lang2, 1)
    else:
        file_base_name = 'clusters'
        all_cluster_sets['word'] = get_clusters(processed_data_dir, file_base_name, lang1, lang2, 0)
        all_cluster_sets['ipa'] = get_clusters(processed_data_dir, file_base_name, lang1, lang2, 1)
        all_cluster_sets['reconstructed'] = get_clusters(processed_data_dir, file_base_name, lang1, lang2, 2)

    g2p_map_train, ed_ceiling_train = make_g2p_map(processed_data_dir, lang2, False)
    g2p_map_test, ed_ceiling_test = make_g2p_map(processed_data_dir, lang2, True)

    print('calculating train vectors...')
    train_vectors = []
    with open(train_data_path, encoding='utf-8') as train_file, open(train_feat_vec_path, 'w+', encoding='utf-8') as train_out:
            for line in train_file:
                print(train_data_path, line)
                vector = []
                graph_rep = line.strip()
                phone_rep = g2p_map_train[graph_rep]
                for tipe, cluster_set in all_cluster_sets.items():
                    if tipe == 'ipa':
                        min_ed = minimize_edit_distance(cluster_set, phone_rep) if phone_rep else ed_ceiling_train
                        vector.append(min_ed)
                        max_cs = maximize_common_substring(cluster_set, phone_rep) if phone_rep else 0
                        vector.append(max_cs)
                    else:
                        min_ed = minimize_edit_distance(cluster_set, graph_rep)
                        vector.append(min_ed)
                        max_cs = maximize_common_substring(cluster_set, graph_rep)
                        vector.append(max_cs)
                train_vectors.append(vector)
            print('writing train vectors to file...')    
            train_out.write('\n'.join(train_vectors))
        # print('calculating test vectors for {} clusters...'.format(tipe))
        # test_feat_vecs = calculate_feat_vecs(test_data_path,)
        # print('writing vectors to file...')
        # with open(test_feat_vec_path, encoding='utf-8') as test_out:
        #     test_out.write('\n'.join(test_feat_vecs))