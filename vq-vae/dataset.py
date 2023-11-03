import math
import torch
from collections import defaultdict, Counter
from smart_open import open


def zero(): return 8


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, path=None, vocab=None, random=False, split=False, remove_long_words=False):
        self.capitalize_utf = '^'.encode("utf-8")
        if path is not None:
            self.words, self.freqs = zip(*[
                line.strip().split('\t')
                for line in open(path)
                if line.strip().split('\t')[0].strip() not in ["[UNK]", "[PAR]"] and line.strip().split('\t')[1] != "1"
            ])
            self.words = [self.encode(w.strip())[0] for w in self.words]
            self.freqs = [int(f.strip()) for f in self.freqs]

            if remove_long_words:
                n_words = len(self.words)
                self.words, self.freqs = map(lambda a: list(a), zip(*[(w, f) for w, f in zip(self.words, self.freqs) if len(w) <= args.max_length - 4]))
                print(f"removed {n_words - len(self.words)} words")

            self.freqs_max = max(self.freqs)

        self.random = random
        self.do_split = split
        self.max_length = args.max_length
        self.skipped_indices = 5  # [PAD] and [BOS]

        if vocab:
            self.vocab = vocab
        else:
            assert path is not None
            counter = Counter()
            for word, freq in zip(self.words, self.freqs):
                for char in word:
                    counter[char] += freq
            self.vocab = ["[1]", "[2]", "[3]", "[PAD]", "[BOS]", "[EOS]", "[BOW]", "[EOW]", "[UNK]"] + [c for c, f in counter.most_common() if f > 0]
            print(f"Created a vocabulary of size: {len(self.vocab)}")

        self.char_to_index = defaultdict(zero)
        for i, c in enumerate(self.vocab):
            self.char_to_index[c] = i
        self.index_to_char = self.vocab

        # [BOS], [BOW], `^`, 'h', 'e', 'l', 'l', 'o', [EOW], [EOS]
        self.unk_id = self.char_to_index["[UNK]"]
        self.pad_id = self.char_to_index["[PAD]"]
        self.bow_id = self.char_to_index["[BOW]"]
        self.eow_id = self.char_to_index["[EOW]"]
        self.bos_id = self.char_to_index["[BOS]"]
        self.eos_id = self.char_to_index["[EOS]"]
        self.capitalize_id = self.char_to_index[ord('^')]
        self.continuation_bytes = {i for i, b in enumerate(self.vocab) if b >= 0x80 and b < 0xC0}
        print(f"Number of UTF-8 continuation bytes: {len(self.continuation_bytes)}")

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

    def decode(self, word):
        word = bytes(word).decode("utf-8", errors="ignore")
        if word == "^":
            return word

        return ''.join(
            ch.upper() if i > 0 and word[i-1] == '^' else ch
            for i, ch in enumerate(word)
            if ch != '^'
        )

    def numericalize(self, word, index=0, bow=True, eow=True, prob_keep=None):
        ids = ([self.bow_id] if bow else []) + [self.char_to_index[c] for c in word] + ([self.eow_id] if eow else [])
        ids = self.split(ids, prob_keep, None if self.random else torch.Generator().manual_seed(index))
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def get_collate_fn(self, max_length: int):
        return CollateFunctor(self.pad_id, max_length)

    def ids_to_word(self, ids, ignore_set=None):
        chars = [self.index_to_char[i] for i in ids if i != self.pad_id]

        word, buffer = "", []
        for c in chars:
            if ignore_set is not None and c in ignore_set:
                continue

            if c in ["[UNK]", "[PAD]", "[BOW]", "[EOW]", "[BOS]", "[EOS]"]:
                if len(buffer) > 0:
                    word += self.decode(buffer)
                    buffer = []
                word += c
            else:
                buffer.append(c)

        if len(buffer):
            word += self.decode(buffer)

        return word

    def split(self, word, prob_keep=None, generator=None):
        if self.do_split and word[0] == self.bow_id and word[-1] == self.eow_id and torch.rand([], generator=generator).item() < 0.1:  # remove BOW/EOW
            p = torch.rand([], generator=generator).item()
            if p > 2 / 3:
                word = word[1:-1]
            elif p > 1 / 3:
                word = word[1:]
            else:
                word = word[:-1]

        if len([c for c in word if c not in [self.bow_id, self.eow_id, self.capitalize_id] and c not in self.continuation_bytes]) <= 1:
            return word

        if len(word) <= self.max_length - 2 and (not self.do_split or torch.rand([], generator=generator).item() < prob_keep):
            return word

        if torch.rand([], generator=generator).item() < 0.5:  # keep prefix
            p_split = [
                1.0 if c not in [self.bow_id, self.capitalize_id] and c_next not in self.continuation_bytes else 0.0
                for c, c_next in zip(word, word[1:] + [0])
            ]
            i_split = torch.multinomial(torch.tensor(p_split[:-1]), 1, generator=generator)[0].item()
            split = word[:i_split + 1]
        else:  # keep suffix
            p_split = [
                1.0 if c not in [self.eow_id, self.capitalize_id] and c_next not in self.continuation_bytes else 0.0
                for c, c_next in zip(word, word[1:] + [0])
            ]
            p_split[-2] = p_split[-2] * p_split[-1]
            i_split = torch.multinomial(torch.tensor(p_split[:-1]), 1, generator=generator)[0].item()
            split = word[i_split + 1:]

        return self.split(split, 0.5, generator)

    def __getitem__(self, index):
        prob_keep = math.log(1.0 + self.freqs[index]) / math.log(1.0 + self.freqs_max)
#        prob_keep = math.log(self.freqs[index] + 1) / math.log(self.freqs_max)
        return self.numericalize(self.words[index], index, prob_keep=prob_keep), self.freqs[index]

    def __len__(self):
        return len(self.words)


class CollateFunctor:
    def __init__(self, pad_id, max_length):
        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, batch):
        words, freqs = zip(*batch)
        max_len = max(len(word) for word in words)
        words = [word + [self.pad_id] * (max_len - len(word)) for word in words]
        words = torch.tensor(words)
        freqs = torch.tensor(freqs)
        return words, freqs
