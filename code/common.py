import argparse
import json
import re
import os
from urllib.request import urlretrieve
from pathlib import Path

import numpy as np
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from pyarrow.parquet import ParquetFile
from rouge_score.rouge_scorer import RougeScorer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SEED = 24
PUNKT = PunktParameters()
PUNKT.abbrev_types = {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'jr', 'sr', 'u.k', 'u.s', 'a.m', 'p.m'}
SENTENCE_TOKENIZER = PunktSentenceTokenizer(PUNKT)
ROUGE = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def sentences(text):
    return [sentence.strip() for paragraph in text.splitlines()
            for sentence in SENTENCE_TOKENIZER.tokenize(paragraph) if words(sentence)]


def data(args, split):
    if args.data_dir:
        local = args.data_dir / f'{split}.parquet'
    else:
        cache = Path(os.environ.get('HF_HOME', '.cache')) / 'xsum-parquet'
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / f'{split}.parquet'
        if not local.exists():
            url = ('https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/'
                   f'refs%2Fconvert%2Fparquet/default/{split}/0000.parquet')
            temporary = local.with_suffix('.download')
            urlretrieve(url, temporary)
            temporary.replace(local)
    parquet = ParquetFile(local)
    total = parquet.metadata.num_rows
    limit = getattr(args, split)
    indices = (np.sort(np.random.default_rng(SEED).choice(total, min(limit, total), replace=False))
               if limit else np.arange(total))
    output = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=['document', 'summary', 'id']):
        stop = offset + batch.num_rows
        start_index, end_index = indices.searchsorted([offset, stop])
        if end_index > start_index:
            output.extend(batch.take(indices[start_index:end_index] - offset).to_pylist())
        offset = stop
    return [row for row in output if words(row['document']) and words(row['summary'])]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/final'))
    parser.add_argument('--data-dir', type=Path, help='Directory containing official split Parquet files')
    parser.add_argument('--train', type=int, default=16384)
    parser.add_argument('--validation', type=int, default=240)
    parser.add_argument('--test', type=int, default=600)
    return parser


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def density(source, summary):
    positions = {}
    for index, token in enumerate(source):
        positions.setdefault(token, []).append(index)
    index = total = 0
    while index < len(summary):
        best = 0
        for start in positions.get(summary[index], []):
            length = 0
            while (start + length < len(source) and index + length < len(summary)
                   and source[start + length] == summary[index + length]):
                length += 1
            best = max(best, length)
        total += best ** 2
        index += max(1, best)
    return total / max(1, len(summary))


def metrics(row, summary, extract=None):
    source, candidate = words(row['document']), words(summary)
    result = {key: value.fmeasure for key, value in ROUGE.score(row['summary'], summary).items()}
    result.update(compression=len(candidate) / len(source), density=density(source, candidate))
    if extract is not None:
        content = [word for word in candidate if word not in ENGLISH_STOP_WORDS]
        evidence = set(words(extract))
        result['proxy'] = sum(word in evidence for word in content) / max(1, len(content))
    return result


def evaluate(rows, outputs, extracts=None):
    extracts = [None] * len(rows) if extracts is None else extracts
    scores = [metrics(row, output, extract) for row, output, extract in
              zip(rows, outputs, extracts, strict=True)]
    return {key: sum(score[key] for score in scores) / len(scores) for key in scores[0]}


def examples(rows, outputs):
    return [{'id': row['id'], 'reference': row['summary'], 'prediction': output,
             'metrics': metrics(row, output)} for row, output in zip(rows, outputs, strict=True)]
