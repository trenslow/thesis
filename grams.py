from collections import Counter
import math


def count_ngrams(tokes, en):
    return Counter(zip(*[tokes[i:] for i in range(en)]))


if __name__ == '__main__':
    langs = ['ces', 'pol', 'bul', 'rus']
    for n in range(1, 3):
        for lang in langs:
            print('---calculating ' + lang + ' ' + str(n) + '-grams---')
            training_file = 'training_articles.' + lang
            with open(training_file) as train:
                train_tokens = [token.strip() for token in train.readlines()]
            ngrams = count_ngrams(train_tokens, n)
            if n == 0:
                model = {word: 1.0 / len(ngrams) for word in ngrams}
            elif n == 1:
                total_unigrams = sum(ngrams.values())
                model = {unigram: count / total_unigrams for unigram, count in ngrams.items()}
            else:
                n_minus_1_grams = count_ngrams(train_tokens, n-1)
                model = {ngram: count / n_minus_1_grams[ngram[:-1]] for ngram, count in ngrams.items()}

            with open('test_articles.' + lang) as test:
                test_tokens = [token.strip() for token in test.readlines()]
            test_grams = count_ngrams(test_tokens, n)
            oov_counts = Counter()
            sum_surprisals = 0.0
            for t in test_grams:
                if t not in model:
                    oov_counts.update([t])
                else:
                    prob_word_given_model = model[t]
                    sum_surprisals -= math.log2(prob_word_given_model) * test_grams[t]

            num_oov_tokens = sum(oov_counts.values())
            num_test_tokens = len(test_tokens)
            print('OOV rate : ' + str(num_oov_tokens / num_test_tokens))
            entropy = sum_surprisals / (num_test_tokens - num_oov_tokens)
            perplexity = 2.0**entropy
            print('PP: ' + str(perplexity))