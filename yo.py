from collections import Counter
counter = Counter()
with open('training_articles.pol') as input:
    for line in input:
        word = line.strip()
        counter.update([word])

print(len([word for word, count in counter.items() if count == 1]))
