import json
import pickle

import torch

from common import arguments, data, evaluate, examples, write
from task1.main import extract, graphs
from task2.main import decode, load_model


def run(args):
    torch.set_num_threads(2)
    model, vocab, config = load_model(args.output)
    task1 = json.loads((args.output / 'task1.json').read_text())
    winner = task1['winner']
    threshold = task1['settings'][winner]['threshold']
    k = task1['settings'][winner]['k']
    with (args.output / 'tfidf.pkl').open('rb') as handle:
        vectorizer = pickle.load(handle)
    validation, test = (data(args, split) for split in ('validation', 'test'))

    def generate(split_graphs, count):
        extracts = [extract(graph, threshold, count) for graph in split_graphs]
        return extracts, [decode(model, text, vocab, config) for text in extracts]

    validation_graphs = graphs(vectorizer, validation)
    trials = {}
    for count in (k + 1, k + 3, k + 5):
        _, outputs = generate(validation_graphs, count)
        trials[count] = evaluate(validation, outputs)['rougeL']
    m = max(trials, key=trials.get)
    extracts, outputs = generate(graphs(vectorizer, test), m)
    results = {'Pipeline': evaluate(test, outputs, extracts)}
    saved = {}
    for method, filename in [('TextRank', 'TextRank'), ('LexRank', 'LexRank'),
                             ('Pointer-generator', 'pointer')]:
        rows = json.loads((args.output / f'{filename}.json').read_text())
        if [row['id'] for row in rows] != [row['id'] for row in test]:
            raise ValueError('All tasks must use the same ordered test IDs.')
        saved[method] = [row['prediction'] for row in rows]
        results[method] = evaluate(test, saved[method], extracts)
    write(args.output / 'task3.json', {'m': m, 'validation': trials, 'test': results})
    details = examples(test, outputs)
    for index, detail in enumerate(details):
        detail['extract'] = extracts[index]
        detail['pointer'] = saved['Pointer-generator'][index]
        detail['source'] = test[index]['document']
    write(args.output / 'pipeline.json', details)
    print(results, flush=True)


if __name__ == '__main__':
    run(arguments().parse_args())
