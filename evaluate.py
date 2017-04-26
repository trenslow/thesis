import math
from collections import Counter
from decimal import Decimal
import operator


def read_clusters(l1, l2, g2p):
    clusters = {}
    pwc = {}

    if g2p:
        c = open('clusters_g2p.' + l1 + '.' + l2, encoding='utf-8')
    else:
        c = open('clusters.' + l1 + '.' + l2, encoding='utf-8')
    cluster = ''
    for line in c:
        stripped = line.strip()
        if ':' in stripped:
            cluster = stripped.replace(':', '')
            clusters[cluster] = {}
        else:
            words = stripped.split(',')
            for word in words:
                if word:
                    if len(word.split()) == 2:
                        token, count = word.split()
                        clusters[cluster][token] = Decimal(count)
                    else:
                        clusters[''][''] = Decimal(word)

    c.close()
    sum_mapped_words = 0
    # outfile = open(l1 + '_' + l2 + '.classes', 'w+')
    for clust, tokens in sorted(clusters.items(), key=operator.itemgetter(0)):
        cluster_count = sum(tokens.values())
        num_tokens = len(tokens)
        sum_mapped_words += num_tokens
        # outfile.write(c + ' has ' + str(num_tokens) + ' word(s) mapped to it' + '\n')
        for toke, cnt in tokens.items():
            pwc[toke] = (cnt / cluster_count, clust)
    # outfile.write('avg:' + str(sum_mapped_words / 100))
    # outfile.close()
    return pwc


def read_vocab(lang, g2p):
    vocab = {}
    if g2p:
        voc = open('vocab_g2p.' + lang)
    else:
        voc = open('vocab.' + lang)

    for line in voc.readlines()[:100]:
        ln = line.strip().split()
        if len(ln) == 2:
            clust, count = ln[0], ln[1]
            vocab[clust] = Decimal(count)
        else:
            vocab[''] = Decimal(ln[0])
    voc.close()

    class_count = sum(vocab.values())
    return {cl: co / class_count for cl, co in vocab.items()}

if __name__ == '__main__':
    lang1, lang2 = 'bul', 'rus'
    g2p = True
    pwc = read_clusters(lang1, lang2, g2p)
    pc = read_vocab(lang1, g2p)
    oov_words = Counter()
    surprisal_by_token = {}
    test_tokens = []

    if g2p:
        test = open('test_articles_g2p.' + lang2, encoding='utf-8')
    else:
        test = open('test_articles.' + lang2, encoding='utf-8')

    sum_surprisals = 0.0
    for line in test:
        if g2p:
            token = ''.join(symb for symb in line.strip().split()[1:])
        else:
            token = line.strip()
        test_tokens.append(token)
        if token in pwc:
            prob_wc = pwc[token][0]
            clas = pwc[token][1]
            prob_c = pc[clas]
            surprisal = -math.log(prob_wc, 2) - math.log(prob_c, 2)
            surprisal_by_token[token] = (surprisal, clas)
            sum_surprisals += surprisal
        else:
            oov_words.update([token])
    test.close()

    total_tokens = len(test_tokens)
    entropy = sum_surprisals / total_tokens
    perplexity = math.pow(2, entropy)
    print('for the language pair ' + lang1 + '/' + lang2 + ':')
    print('perplexity =', perplexity)
    print('oov rate =', sum(oov_words.values()) / total_tokens)
    print('total test tokens = ', total_tokens)

    if g2p:
        out = open('surprisal_g2p.' + lang1 + '.' + lang2, 'w+')
    else:
        out = open('surprisal.' + lang1 + '.' + lang2, 'w+')

    for token, double in sorted(surprisal_by_token.items(), key=operator.itemgetter(0)):
        out.write(token + ' ' + str(double[0]) + ' ' + double[1] + '\n')
    out.close()