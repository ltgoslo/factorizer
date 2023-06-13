from functools import lru_cache
from typing import Union, NamedTuple, List, Tuple
import torch
import numpy as np
import dawg


class Encoding(NamedTuple):
    ids: List[Tuple[int, int, int]]
    tokens: List[str]
    perplexities: List[float]
    offsets: List[Tuple[int, int]]


def whitespace_split(line):
    words = line.split()
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        running_offset = word_offset + len(word)
        append((word, word_offset, running_offset))

    return offsets


class Node:
    def __init__(self, suffix, cost, predecessor, indices):
        self.suffix = suffix
        self.cost = cost
        self.predecessor = predecessor
        self.indices = indices
        self.is_open = True


@lru_cache(maxsize=1024)
def char_to_bytes(ch: str):
    encoded = ch.encode("utf-8")
    encoded = [c for ch in encoded for c in ([0xC0 | (ch >> 3), 0x80 | (ch & 0x07)] if ch >= 0x80 else [ch])]
    decoded = bytes(encoded).decode("utf-8", errors="strict")
    return decoded


@lru_cache(maxsize=1024)
def bytes_to_subword(b: str):
    encoded = b.encode("utf-8")
    encoded = [
        c if c < 0x80 else ((encoded[i - 1] << 3) & 0xFF) | (c & 0x07)
        for i, c in enumerate(encoded)
        if c < 0xC0
    ]
    decoded = bytes(encoded).decode("utf-8", errors="strict")
    return decoded


def optimal_search(trie, word, alpha=1.0, beta=0.0, add_bow=True, add_eow=True, sample=False):
    alignment, encoded_word = [], []

    if add_bow:
        w = char_to_bytes('⸥')
        encoded_word.append(w)
        alignment += [0] * len(w)

    for i, ch in enumerate(word):
        w = char_to_bytes(ch)
        encoded_word.append(w)
        alignment += [i] * len(w)
    
    if add_eow:
        w = char_to_bytes('⸤')
        encoded_word.append(w)
        alignment += [len(word) - 1] * len(w)

    alignment += [len(word)]
    encoded_word = ''.join(encoded_word)

    if beta == 0.0:
        return dijkstra(trie, encoded_word, alignment, alpha)

    for _ in range(16):
        encoding = dijkstra_beta(trie, encoded_word, alignment, alpha, beta, sample=sample)
        if encoding[1][0].startswith("⸥") and encoding[1][-1].endswith("⸤"):
            return encoding

    return dijkstra(trie, word, alignment, alpha)


class OOVException(Exception):
    pass


def dijkstra_beta(trie, word, alignment, alpha, beta, sample: True):
    if len(word) > 32:
        raise OOVException()

    initial_node = Node(word, 0.0, None, None)
    open_nodes = [initial_node]
    all_nodes = {initial_node.suffix: initial_node}

    while len(open_nodes) > 0:
        min_node = open_nodes.pop(0)
        min_node.is_open = False

        if len(min_node.suffix) == 0:
            break

        if '' in all_nodes and min_node.cost + alpha >= all_nodes[''].cost:
            min_node = all_nodes['']
            break

        for prefix in trie.prefixes(min_node.suffix)[::-1]:
            noise = beta * len(prefix) * (torch.randn([]).exp().item())
            suffix = min_node.suffix[len(prefix):]

            if suffix != '' and '' in all_nodes and min_node.cost + noise + 2*alpha >= all_nodes[''].cost:
                continue

            if suffix in all_nodes:
                old_node = all_nodes[suffix]
                if min_node.cost + noise + alpha >= old_node.cost:
                    continue

                values = trie.get_value(prefix)
                perplexity, *indices = max(values, key=lambda item: item[0])
                if min_node.cost - perplexity + noise + alpha >= old_node.cost:
                    continue
                if suffix != '' and '' in all_nodes and min_node.cost - perplexity + noise + 2*alpha >= all_nodes[''].cost:
                    continue

            else:
                values = trie.get_value(prefix)
                perplexity, *indices = max(values, key=lambda item: item[0])

            if sample and len(values) > 1:
                perplexities = [p for p, *_ in values]
                probs = 1.0 / (torch.tensor(perplexities) - 1e-12)
                probs = probs / probs.sum()
                index = probs.multinomial(1).item()
                perplexity, *indices = values[index]

            cost = min_node.cost + alpha - perplexity + noise

            if suffix != '' and '' in all_nodes and cost + alpha >= all_nodes[''].cost:
                continue

            if suffix not in all_nodes:
                new_node = Node(suffix, cost, min_node, indices)
                open_nodes.append(new_node)
                all_nodes[suffix] = new_node
            elif cost < old_node.cost:
                assert old_node.is_open
                old_node.cost, old_node.predecessor, old_node.indices = cost, min_node, indices

        open_nodes = sorted(open_nodes, key=lambda node: node.cost)

    if len(min_node.suffix) != 0:
        raise OOVException()

    indices, subwords, perplexities, offsets, offset = [], [], [], [], 1
    node = min_node
    while node.predecessor is not None:
        indices.append(node.indices)
        perplexities.append(node.cost)

        subword = node.predecessor.suffix[:len(node.predecessor.suffix)-len(node.suffix)]
        offsets.append((alignment[-(offset + len(subword))], alignment[-offset]))
        offset += len(subword)

        try:
            subword = bytes_to_subword(subword)
        except:
            pass

        subwords.append(subword)
        node = node.predecessor

    return indices[::-1], subwords[::-1], perplexities[::-1], offsets[::-1]


def dijkstra(trie, word, alignment, alpha):
    if len(word) > 32:
        raise OOVException()

    initial_node = Node(word, 0.0, None, None)
    open_nodes = [initial_node]
    all_nodes = {initial_node.suffix: initial_node}

    while len(open_nodes) > 0:
        min_node = open_nodes.pop(0)
        min_node.is_open = False

        if len(min_node.suffix) == 0:
            break

        if '' in all_nodes and min_node.cost + alpha >= all_nodes[''].cost:
            min_node = all_nodes['']
            break
    
        for prefix in trie.prefixes(min_node.suffix)[::-1]:
            perplexity, *indices = max(trie.get_value(prefix), key=lambda item: item[0])
            # perplexity, *indices = trie.get_value(prefix)[0]

            suffix = min_node.suffix[len(prefix):]
            cost = min_node.cost + alpha - perplexity

            if suffix != '' and '' in all_nodes and cost + alpha >= all_nodes[''].cost:
                continue

            if suffix not in all_nodes:
                new_node = Node(suffix, cost, min_node, indices)
                open_nodes.append(new_node)
                all_nodes[suffix] = new_node
            elif cost < (old_node := all_nodes[suffix]).cost:
                assert old_node.is_open
                old_node.cost, old_node.predecessor, old_node.indices = cost, min_node, indices

        open_nodes = sorted(open_nodes, key=lambda node: node.cost)

    if len(min_node.suffix) != 0:
        raise OOVException()

    indices, subwords, perplexities, offsets, offset = [], [], [], [], 1
    node = min_node
    while node.predecessor is not None:
        indices.append(node.indices)
        perplexities.append(node.cost)

        subword = node.predecessor.suffix[:len(node.predecessor.suffix)-len(node.suffix)]
        offsets.append((alignment[-(offset + len(subword))], alignment[-offset]))
        offset += len(subword)

        try:
            subword = bytes_to_subword(subword)
        except:
            pass

        subwords.append(subword)
        node = node.predecessor

    return indices[::-1], subwords[::-1], perplexities[::-1], offsets[::-1]


class Factorizer:
    def __init__(self, tokenizer_path: str, alpha=1.0, beta=0.0, merge_unks=True, allow_decoding=False, sample=False):
        self.special_id_to_token = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]", "[SPECIAL]"]
        self.special_token_to_id = {token: i for i, token in enumerate(self.special_id_to_token)}
        self.unk_id = (self.special_token_to_id["[UNK]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.cls_id = (self.special_token_to_id["[CLS]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.sep_id = (self.special_token_to_id["[SEP]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.pad_id = (self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.mask_id = (self.special_token_to_id["[MASK]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.sos_id = (self.special_token_to_id["[SOS]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.eos_id = (self.special_token_to_id["[EOS]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])
        self.special_id = (self.special_token_to_id["[SPECIAL]"], self.special_token_to_id["[PAD]"], self.special_token_to_id["[PAD]"])

        self.n_special_tokens = len(self.special_token_to_id)
        self.vocab_size = 256 + self.n_special_tokens

        self.trie = dawg.RecordDAWG("fBBB", payload_separator=b'\xff').load(tokenizer_path)

        self.alpha = alpha
        self.beta = beta
        self.merge_unks = merge_unks
        self.sample = sample

        self.allow_decoding = allow_decoding
        if allow_decoding:
            self.load_inverse_vocab()

    def __call__(self, text: Union[str, List[str]]):
        return self.encode(text)

    def load_inverse_vocab(self):
        self.id_to_subword = np.zeros((256, 256, 256), dtype=object)
        for subword, (_, index_1, index_2, index_3) in self.trie.iteritems():
            self.id_to_subword[index_1, index_2, index_3] = bytes_to_subword(subword)

    def encode(self, text: Union[str, List[str]]) -> Union[Encoding, List[Encoding]]:
        if isinstance(text, (list, tuple)):
            return [self.encode(t) for t in text]
        
        assert isinstance(text, str), f"Expected str, got {type(text)}"

        ids, subwords, perplexities, offsets = [], [], [], []
        for word, start, end in whitespace_split(text):
            if self.beta == 0.0:
                output = self.tokenize_word_cached(word)
            else:
                output = self.tokenize_word(word)

            ids += output.ids
            subwords += output.tokens
            perplexities += output.perplexities
            offsets += [(start + start_, start + end_) for (start_, end_) in output.offsets]

            if end != offsets[-1][1]:
                print(f"ERROR in offseting {text} -> {' '.join(subwords)}", flush=True)

        return Encoding(ids, subwords, perplexities, offsets)

    def decode(self, indices: Union[List[Tuple[int]], List[List[Tuple[int]]]], skip_special_tokens=True) -> Union[str, List[str]]:
        if not self.allow_decoding:
            self.load_inverse_vocab()
            self.allow_decoding = True

        assert isinstance(indices, (list, tuple)), f"Expected list, got {type(indices)}"
        assert isinstance(indices[0], (list, tuple)), f"Expected list of tuples, got list of {type(indices[0])}"

        if isinstance(indices[0][0], (list, tuple)):
            return [self.decode(index) for index in indices]
        
        assert all(len(index) == 3 for index in indices), f"Expected list of tuples of length 3"
    
        output = []
        for index_1, index_2, index_3 in indices:
            if index_1 < self.n_special_tokens or index_2 < self.n_special_tokens or index_3 < self.n_special_tokens:
                if not skip_special_tokens:
                    subword = f" {self.special_id_to_token[index_1]} "
                    output.append(subword)
                continue

            subword = self.id_to_subword[index_1 - self.n_special_tokens, index_2 - self.n_special_tokens, index_3 - self.n_special_tokens]
            if subword == 0:
                output.append(" [UNK] ")
                continue

            subword = subword.replace("⸥", " ").replace("⸤", " ")
            output.append(subword)

        return ' '.join(''.join(output).split())

    @lru_cache(maxsize=65536)
    def tokenize_word_cached(self, word: str, add_bow=True, add_eow=True):
        return self.tokenize_word(word, add_bow, add_eow)

    def tokenize_word(self, word: str, add_bow=True, add_eow=True):
        if word.lower() == "[unk]":
            return Encoding([self.unk_id], [word], [float("-inf")], [(0, len(word))])

        try:
            result = optimal_search(self.trie, word, self.alpha, self.beta, add_bow, add_eow, self.sample)
        except OOVException:
            return Encoding([self.unk_id], [word], [float("-inf")], [(0, len(word))])

        ids, subwords, perplexities, offsets = result

        for i, subword in enumerate(subwords):
            ids[i] = (ids[i][0] + self.n_special_tokens, ids[i][1] + self.n_special_tokens, ids[i][2] + self.n_special_tokens)
            if "[unk]" in subword.lower():
                if self.merge_unks:
                    return Encoding([self.unk_id], ["[UNK]"], [float("-inf")], [(0, len(word))])
                else:
                    ids[i] = self.unk_id

        return Encoding(ids, subwords, perplexities, offsets)
