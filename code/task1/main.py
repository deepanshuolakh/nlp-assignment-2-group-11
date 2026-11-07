import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import arguments, data, evaluate, examples, sentences, write


def rank(similarity, threshold):
    weights = np.where(similarity >= threshold, similarity, 0)
    size = len(weights)
    degree = weights.sum(axis=1, keepdims=True)
    transition = np.divide(weights, degree, out=np.full_like(weights, 1 / size), where=degree > 0)
    scores = np.full(size, 1 / size)
    for _ in range(100):
        updated = 0.15 / size + 0.85 * (scores @ transition)
        if np.abs(updated - scores).sum() < 1e-9:
            break
        scores = updated
    return np.argsort(-updated, kind='stable')


def graphs(vectorizer, rows):
    result = []
    for row in rows:
        units = sentences(row['document'])
        vectors = vectorizer.transform(units)
        similarity = (vectors @ vectors.T).toarray()
        np.fill_diagonal(similarity, 0)
        result.append((units, similarity))
    return result


def extract(graph, threshold, count):
    units, similarity = graph
    order = rank(similarity, threshold)
    return ' '.join(units[index] for index in sorted(order[:count]))


def run(args):
    train, val, test = (data(args, split) for split in ('train', 'validation', 'test'))
    vectorizer = TfidfVectorizer(max_features=50000, token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(unit for row in train for unit in sentences(row['document']))
    validation_graphs = graphs(vectorizer, val)
    test_graphs = graphs(vectorizer, test)
    settings = {}
    results = {}
    for name, thresholds in [('TextRank', [0.0]), ('LexRank', [0.1, 0.2, 0.3])]:
        trials = []
        for threshold in thresholds:
            for k in (1, 2, 3):
                outputs = [extract(graph, threshold, k) for graph in validation_graphs]
                score = evaluate(val, outputs)['rougeL']
                trials.append((score, threshold, k))
        score, threshold, k = max(trials, key=lambda trial: trial[0])
        settings[name] = {'rougeL': score, 'threshold': threshold, 'k': k, 'trials': trials}
        outputs = [extract(graph, threshold, k) for graph in test_graphs]
        results[name] = evaluate(test, outputs)
        write(args.output / f'{name}.json', examples(test, outputs))
    winner = max(settings, key=lambda name: settings[name]['rougeL'])
    write(args.output / 'task1.json', {'winner': winner, 'settings': settings, 'test': results,
                                     'sizes': [len(train), len(val), len(test)]})
    with (args.output / 'tfidf.pkl').open('wb') as handle:
        pickle.dump(vectorizer, handle)
    print(results, flush=True)


if __name__ == '__main__':
    run(arguments().parse_args())
