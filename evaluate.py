import math
from collections import Counter
import operator


def read_clusters(l1, l2, g2p):
    clusters = {}
    pwc = {}
    if g2p:
        cluster_file = open('clusters_g2p.' + l1 + '.' + l2, encoding='utf-8')
    else:
        cluster_file = open('clusters.' + l1 + '.' + l2, encoding='utf-8')

    cluster = ''
    for line in cluster_file:
        stripped = line.strip()
        if ':' in stripped:
            cluster = stripped[:-1]
            clusters[cluster] = {}
        else:
            words = stripped.split(',')
            if len(words[0]) > 0:
                for word in words:
                    split = word.split()
                    if len(split) == 1:
                        clusters[cluster][''] = float(split[0])
                    else:
                        token, count = word.split()
                        clusters[cluster][token] = float(count)
                    
    cluster_file.close()

    sum_mapped_words = 0.0
    for c, tokens in sorted(clusters.items(), key=operator.itemgetter(0)):
        cluster_count = sum(tokens.values())
        num_tokens = len(tokens)
        sum_mapped_words += num_tokens
        for toke, cnt in tokens.items():
            pwc[toke] = (cnt / cluster_count, c)

    return pwc


def read_vocab(lang, g2p):
    vocab = {}
    if g2p:
        voc = open('vocab_g2p.' + lang)
    else:
        voc = open('vocab.' + lang)
    for line in voc.readlines()[:100]:
        split = line.strip().split()
        if len(split) == 1:
            vocab[''] = float(split[0])
        else:
            clust, count = split
            vocab[clust] = float(count)
    voc.close()
    class_count = sum(vocab.values())
    return {cl: co / class_count for cl, co in vocab.items()}


if __name__ == '__main__':
    lang1, lang2 = 'ces', 'pol'
    g2p = True
    pwc = read_clusters(lang1, lang2, g2p)
    pc = read_vocab(lang1, g2p)
    oov_words = Counter()
    surprisal_by_token = {}
    num_test_tokens = 0

    if g2p:
        test = open('test_articles_g2p.' + lang2, encoding='utf-8')
    else:
        test = open('test_articles.' + lang2, encoding='utf-8')

    sum_surprisals = 0.0
    for line in test:
        num_test_tokens += 1
        if g2p:
            token = ''.join(symb for symb in line.strip().split()[1:])
        else:
            token = line.strip()
        if token in pwc:
            prob_wc = pwc[token][0]
            clas = pwc[token][1]
            prob_c = pc[clas]
            surprisal = -math.log2(prob_c) - math.log2(prob_wc)
            surprisal_by_token[token] = (surprisal, clas)
            sum_surprisals += surprisal
        else:
            oov_words.update([token])
    test.close()

    entropy = sum_surprisals / num_test_tokens
    perplexity = 2**entropy
    print('for the language pair ' + lang1 + '/' + lang2 + ':')
    print('perplexity =', perplexity)
    print('oov rate =', sum(oov_words.values()) / num_test_tokens)
    print('number of test tokens = ', num_test_tokens)

    with open('surprisal.' + lang1 + '.' + lang2, 'w+') as out:
        for token, double in sorted(surprisal_by_token.items(), key=operator.itemgetter(0)):
            out.write(token + ' ' + str(double[0]) + ' ' + double[1] + '\n')
