import torch
import gzip
import pickle
import dawg
from tqdm import tqdm
from collections import defaultdict
import sys

from model import Model
from dataset import Dataset


class CharEncoder:
    def __init__(self, vocab):
        self.capitalize_utf = '^'.encode("utf-8")
        self.vocab = vocab

        self.char_to_index = defaultdict(int, {c: i for i, c in enumerate(self.vocab)})
        self.index_to_char = self.vocab

        self.unk_id = self.char_to_index["[UNK]"]
        self.pad_id = self.char_to_index["[PAD]"]
        self.bow_id = self.char_to_index["[BOW]"]
        self.eow_id = self.char_to_index["[EOW]"]
        self.bos_id = self.char_to_index["[BOS]"]
        self.eos_id = self.char_to_index["[EOS]"]
        self.capitalize_id = self.char_to_index[ord('^')]

    def encode(self, word: str):
        utf_string, alignment = [], []
        for i, c in enumerate(word):
            if c.isupper():
                utf_string.append(self.capitalize_utf)
                alignment.append(i)

            utf_c = c.lower().encode("utf-8")
            utf_string.append(utf_c)
            alignment += len(utf_c) * [i]

        return b''.join(utf_string), alignment

    def decode(self, word, errors="ignore"):
        word = bytes(word).decode("utf-8", errors=errors)
        if word == "^":
            return word

        return ''.join(
            ch.upper() if i > 0 and word[i-1] == '^' else ch
            for i, ch in enumerate(word)
            if ch != '^'
        )

    def ids_to_word(self, ids, ignore_set=None, errors="ignore"):
        chars = [self.index_to_char[i] for i in ids if i != self.pad_id]

        word, buffer = "", []
        for c in chars:
            if ignore_set is not None and c in ignore_set:
                continue

            if c in ["[UNK]", "[PAD]", "[BOW]", "[EOW]", "[BOS]", "[EOS]"]:
                if len(buffer) > 0:
                    word += self.decode(buffer, errors=errors)
                    buffer = []
                word += c
            else:
                buffer.append(c)

        if len(buffer):
            word += self.decode(buffer, errors=errors)

        return word


def create_vocabulary(args, dataset, checkpoint, filename: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    pad_id = dataset.char_to_index["[PAD]"]
    unk_id = dataset.char_to_index["[UNK]"]
    bos_id = dataset.char_to_index["[BOS]"]
    eos_id = dataset.char_to_index["[EOS]"]

    model = Model(args, dataset)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    vocabulary = []

    index_count = checkpoint["index_count"]
    index_p = index_count / index_count.sum()
    index_logp = index_p.log()

    print(index_logp.shape)
    print(index_logp.max(), index_logp.min(), index_logp.mean(), flush=True)

    for i in tqdm(range(256)):
        chunk = 16
        for j in range(0, 256, chunk):
            indices_2 = torch.stack([
                torch.full([chunk], i, device=device),
                torch.arange(j, j+chunk, device=device),
                torch.full([chunk], 0, device=device)
            ], dim=1)
            indices_3 = torch.stack([
                torch.full([256], 0, device=device),
                torch.full([256], 0, device=device),
                torch.arange(0, 256, device=device)
            ], dim=1)
            indices = (indices_2.unsqueeze(1) + indices_3.unsqueeze(0)).flatten(0, 1)

            predictions, perplexities = model.decode(indices)
            prior_p = index_logp[i * (256*256) + j * 256 : i * (256*256) + (j+chunk) * 256]
            vocabulary += list(zip(predictions, perplexities, prior_p.tolist()))

            print(f"{i} / 256, {j} / 256, {len(vocabulary)}", flush=True)

    with gzip.open(filename, mode='wb') as f:
        pickle.dump(vocabulary, f, protocol=-1)

    return vocabulary


def create_trie(charmap, vocabulary, filename: str):
    char_encoder = CharEncoder(charmap)

    zero_indices = 0
    unicode_errors = 0

    words = {}
    for index, (word, perplexity, prior_logprob) in enumerate(vocabulary):
        if prior_logprob == float("-inf"):
            zero_indices += 1
            continue

        try:
            word = char_encoder.ids_to_word(tuple(word), errors="strict")
        except UnicodeDecodeError:
            unicode_errors += 1
            continue

        index = (index // (256*256), (index // 256) % 256, index % 256)
        score = perplexity + prior_logprob

        if word not in words or score > words[word][1]:
            words[word] = (index, score)

    print(f"Zero indices: {zero_indices / (256*256*256) * 100.0:.2f}%")
    print(f"Unicode errors: {unicode_errors / (256*256*256) * 100.0:.2f}%")

    words = [(word, (perplexity, *index)) for word, (index, perplexity) in words.items()]

    vocabulary = sorted(words, key=lambda item: tuple(item[0]))
    print(len(vocabulary))

    trie = dawg.RecordDAWG("fBBB", vocabulary, payload_separator=b'\xff')
    # trie = marisa_trie.RecordTrie("fBBB", vocabulary)
    trie.save(filename)

    return trie


if __name__ == "__main__":
    checkpoint = torch.load("../czech_256_mog_12345.bin", map_location="cpu")

    args = checkpoint["args"]
    dataset = Dataset(args, path=None, vocab=checkpoint["vocabulary"])
    dataset.split_every = float("inf")

    vocab_filename = "czech_123.pickle.gz"
    trie_filename = "czech_prior.dawg"
    char_vocab_filename = "charmap_czech_123.pickle.gz"

    with gzip.open(char_vocab_filename, mode='wb') as f:
        pickle.dump(dataset.vocab, f, protocol=-1)

    vocabulary = create_vocabulary(args, dataset, checkpoint, vocab_filename)

    with gzip.open(char_vocab_filename, mode='rb') as f:
        charmap = pickle.load(f)

    with gzip.open(vocab_filename, mode='rb') as f:
        vocabulary = pickle.load(f)

    trie = create_trie(charmap, vocabulary, trie_filename)
