import nltk
from collections import Counter
import math
import operator

def count_ngrams(file, n):
    with open(file) as train:
        tokens = [token.strip() for token in train.readlines()]
    return Counter(zip(*[tokens[i:] for i in range(n)])), len(tokens)


if __name__ == '__main__':
    langs = ['bul', 'ces', 'pol', 'rus']
    for n in range(1, 4): # my computer barely handle 3, let alone more
        for lang in langs[:]:
            print('---calculating ' + lang + ' ' + str(n) + '-grams---')
            ngrams, num_train_tokens = count_ngrams('training_articles.' + lang, n)
            if n == 0:
                model = {word: 1.0 / len(ngrams) for word in ngrams}
            elif n == 1:
                total_unigrams = sum(ngrams.values())
                model = {unigram: count / total_unigrams for unigram, count in ngrams.items()}
            else:
                n_minus_1_grams, num_n_minus_1_grams = count_ngrams('training_articles.' + lang, n-1)
                model = {ngram: count / n_minus_1_grams[ngram[:-1]] for ngram, count in ngrams.items()}

            test_grams, num_test_tokens = count_ngrams('test_articles.' + lang, n)
            oov_counts = Counter()
            sum_surprisals = 0.0
            for t in test_grams:
                if t not in model:
                    oov_counts.update([t])
                else:
                    prob_word_given_model = model[t]
                    sum_surprisals -= math.log2(prob_word_given_model) * test_grams[t]

            if n == 1:
                num_oov_tokens = sum(oov_counts.values())
                print('OOV rate : ' + str(num_oov_tokens / num_test_tokens))
            entropy = sum_surprisals / (num_test_tokens - num_oov_tokens)
            perplexity = 2.0**entropy
            print('PP: ' + str(perplexity))