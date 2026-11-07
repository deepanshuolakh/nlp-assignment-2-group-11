import time
from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from common import SEED, arguments, data, evaluate, examples, words, write

PAD, UNK, BOS, EOS = range(4)
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']


def encode(source, target, vocab, source_limit, target_limit):
    oov = {}
    size = len(vocab)
    source_words = words(source)[:source_limit] or ['<unk>']
    base, extended = [], []
    for word in source_words:
        base.append(vocab.get(word, UNK))
        extended.append(vocab[word] if word in vocab else oov.setdefault(word, size + len(oov)))
    gold = [vocab.get(word, oov.get(word, UNK)) for word in words(target)[:target_limit - 1]]
    return base, extended, gold + [EOS], list(oov)


def collate(rows):
    fields = [pad_sequence([torch.tensor(row[index]) for row in rows], batch_first=True)
              for index in range(3)]
    return (*fields, max(len(row[3]) for row in rows))


class PointerGenerator(nn.Module):
    def __init__(self, size, embedding, hidden):
        super().__init__()
        self.size = size
        self.embedding = nn.Embedding(size, embedding, padding_idx=PAD)
        self.encoder = nn.LSTM(embedding, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embedding + 2 * hidden, hidden)
        self.init_h = nn.Linear(2 * hidden, hidden)
        self.init_c = nn.Linear(2 * hidden, hidden)
        self.enc_attn = nn.Linear(2 * hidden, hidden, bias=False)
        self.dec_attn = nn.Linear(hidden, hidden)
        self.cov_attn = nn.Linear(1, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)
        self.generate = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, size))
        self.gate = nn.Linear(3 * hidden + embedding, 1)

    def start(self, source):
        mask = source.ne(PAD)
        packed = pack_padded_sequence(self.embedding(source), mask.sum(1), batch_first=True,
                                      enforce_sorted=False)
        packed_out, (h, c) = self.encoder(packed)
        memory, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=source.shape[1])
        state = (self.init_h(torch.cat(tuple(h), dim=1)).tanh(),
                 self.init_c(torch.cat(tuple(c), dim=1)).tanh(),
                 memory.new_zeros((source.shape[0], memory.shape[-1])),
                 memory.new_zeros(source.shape))
        return (memory, self.enc_attn(memory), mask), state

    def step(self, token, state, memory, extended, max_oov):
        h, c, context, coverage = state
        encoded, features, mask = memory
        embedding = self.embedding(token.masked_fill(token >= self.size, UNK))
        h, c = self.decoder(torch.cat((embedding, context), dim=-1), (h, c))
        energy = self.energy((features + self.dec_attn(h).unsqueeze(1)
                              + self.cov_attn(coverage.unsqueeze(-1))).tanh()).squeeze(-1)
        attention = energy.masked_fill(~mask, -torch.inf).softmax(dim=1)
        context = torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)
        combined = torch.cat((h, context), dim=-1)
        gate = self.gate(torch.cat((combined, embedding), dim=-1)).sigmoid()
        distribution = self.generate(combined).softmax(dim=-1) * gate
        distribution = nn.functional.pad(distribution, (0, max_oov))
        distribution = distribution.scatter_add(1, extended, attention * (1 - gate))
        penalty = torch.minimum(attention, coverage).sum(dim=1)
        return distribution, (h, c, context, coverage + attention), penalty

    def loss(self, source, extended, target, max_oov):
        memory, state = self.start(source)
        previous = torch.full_like(target[:, 0], BOS)
        loss = 0
        for label in target.unbind(dim=1):
            probabilities, state, penalty = self.step(previous, state, memory, extended, max_oov)
            nll = -probabilities.gather(1, label[:, None]).squeeze(1).clamp_min(1e-12).log()
            loss += ((nll + penalty) * label.ne(PAD)).sum()
            previous = label
        return loss / target.ne(PAD).sum()


@torch.inference_mode()
def decode(model, source, vocab, config):
    base, extended, _, oov = encode(source, '', vocab, config['source_limit'], config['target_limit'])
    memory, state = model.start(torch.tensor([base]))
    extended = torch.tensor([extended])
    beam = [([BOS], 0.0, state)]
    finished = []

    def score(item):
        return item[1] / (len(item[0]) - 1) ** 0.7

    for index in range(config['target_limit']):
        candidates = []
        for tokens, logp, state in beam:
            distribution, next_state, _ = model.step(torch.tensor([tokens[-1]]), state, memory,
                                                     extended, len(oov))
            scores = distribution[0].clamp_min(1e-12).log()
            scores[[PAD, UNK, BOS]] = -torch.inf
            if index < 4:
                scores[EOS] = -torch.inf
            values, ids = scores.topk(config['beam'])
            for value, token in zip(values.tolist(), ids.tolist()):
                if value == -torch.inf:
                    continue
                item = (tokens + [token], logp + value, next_state)
                (finished if token == EOS else candidates).append(item)
        beam = sorted(candidates, key=score, reverse=True)[:config['beam']]
        if not beam:
            break
    best = max(finished + beam, key=score)[0][1:]
    dictionary = list(vocab) + oov
    return ' '.join(dictionary[token] for token in best if token != EOS)


def load_model(output):
    checkpoint = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
    vocab, config = checkpoint['vocab'], checkpoint['config']
    model = PointerGenerator(len(vocab), config['embedding'], config['hidden'])
    model.load_state_dict(checkpoint['model'])
    return model.eval(), vocab, config


def parser():
    result = arguments()
    for name, default in [('embedding', 48), ('hidden', 48), ('vocab_size', 6000),
                          ('source_limit', 160), ('target_limit', 28), ('beam', 4),
                          ('epochs', 3), ('batch', 32), ('threads', 2)]:
        result.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    result.add_argument('--resume', action='store_true')
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    train = data(args, 'train')
    counts = Counter(word for row in train for field in ('document', 'summary') for word in words(row[field]))
    vocab = {word: index for index, word in enumerate(SPECIAL + [
        word for word, _ in counts.most_common(args.vocab_size - len(SPECIAL))])}
    config = {key: getattr(args, key) for key in ('embedding', 'hidden', 'vocab_size', 'source_limit',
                                                  'target_limit', 'beam', 'batch', 'train')}
    samples = [encode(row['document'], row['summary'], vocab, args.source_limit, args.target_limit)
               for row in train]
    model = PointerGenerator(len(vocab), args.embedding, args.hidden)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15, initial_accumulator_value=0.1)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'model.pt'
    history = []
    if args.resume:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config'] != config or saved['vocab'] != vocab:
            raise ValueError('Resume requires the original training configuration and vocabulary.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']
    for epoch in range(len(history), args.epochs):
        start = time.monotonic()
        model.train()
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(SEED + epoch))
        total = tokens = 0
        for indices in order.split(args.batch):
            batch = collate([samples[index] for index in indices.tolist()])
            optimizer.zero_grad()
            loss = model.loss(*batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = int(batch[2].ne(PAD).sum())
            total += loss.item() * count
            tokens += count
        entry = {'epoch': epoch + 1, 'loss': total / tokens, 'seconds': time.monotonic() - start}
        history.append(entry)
        temporary = path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'vocab': vocab,
                    'config': config, 'history': history}, temporary)
        temporary.replace(path)
        print(entry, flush=True)
    write(args.output / 'training.json', {'config': config, 'history': history, 'train_size': len(train),
                                         'vocab_size': len(vocab)})
    test = data(args, 'test')
    model.eval()
    predictions = [decode(model, row['document'], vocab, config) for row in test]
    results = evaluate(test, predictions)
    write(args.output / 'task2.json', results)
    write(args.output / 'pointer.json', examples(test, predictions))
    print(results, flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
