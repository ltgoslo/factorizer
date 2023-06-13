<h2 align="center"><b><h3>Tokenization with Factorized Subword Encoding</h3></b></h2><br>


<p align="center">
  <b>David Samuel</b> and <b>Lilja Øvrelid</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2303.09859"><b>Paper (arXiv)</b></a><br><br>
</p>

_______

<br>

## Installation

```bash
git clone https://github.com/ltgoslo/factorizer.git
cd factorizer
python3 setup.py install  
```

<br>

## Usage

```python
from factorizer import Factorizer


tokenizer = Factorizer("english.dawg")
sentence = "The echo of a distant time comes willowing across the sand, and everything is green and submarine."

encoding = tokenizer(sentence)

print(f"INPUT:    {sentence}")
print(f"SUBWORDS: {' '.join(encoding.tokens)}")
print(f"INDICES:  {' '.join(str(index) for index in encoding.ids)}")
print(f"DECODED:  {tokenizer.decode(encoding.ids}")
```

This should output:
```
INPUT:    The echo of a distant time comes willowing across the sand, and everything is green and submarine.
SUBWORDS: ⸥The⸤ ⸥echo⸤ ⸥of⸤ ⸥a⸤ ⸥distant⸤ ⸥time⸤ ⸥comes⸤ ⸥wil lowing⸤ ⸥across⸤ ⸥the⸤ ⸥sand ,⸤ ⸥and⸤ ⸥everything⸤ ⸥is⸤ ⸥green⸤ ⸥and⸤ ⸥submarine .⸤
INDICES:  (52, 74, 62) (221, 21, 77) (135, 64, 137) (181, 45, 79) (248, 77, 122) (88, 92, 159) (124, 92, 64) (49, 151, 114) (79, 180, 104) (129, 186, 151) (52, 74, 219) (49, 127, 34) (35, 174, 39) (76, 101, 35) (32, 176, 191) (135, 209, 205) (44, 28, 242) (76, 101, 35) (13, 171, 144) (211, 41, 131)
DECODED:  The echo of a distant time comes willowing across the sand, and everything is green and submarine.
```

<br>

## Documentation

#### class Encoding:

A named tuple containing:
- `ids` (List[Tuple[int, int, int]])
- `tokens` (List[str])
- `perplexities` (List[float])
- `offsets` (List[Tuple[int, int]])

#### `Factorizer.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `tokenizer_path` (str) | path to a DAWG file containing with pretrained vocabulary |
| `alpha` (float) | the alpha_split hyperparameter controling the granularity of subword splits *(default: 0.1)* |
| `sigma` (float)           | the sigma_sample hyperparameter controling the randomness (temperature) of sampling *(default: 0.0)* (no sampling) |
| `merge_unks` (bool)       | set this argument to True if you want to merge consecutive UNK tokens *(default: True)* |
| `allow_decoding` (bool)       | set this argument to True if you want to precompute the inverse vocabulary for decoding *(default: False)* |
| `sample` (bool)       | set this argument to True if you want to sample from the subword distribution; set to False if you want to always do the optimal tokenization *(default: False)* |

<br>

#### `Factorizer.__call__`

Factorizes the input string (or list of strings)

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `text` (Union[str, List[str]]) | the input string (or list of strings) |

**Returns**: Union[Encoding, List[Encoding]]

<br>

#### `Factorizer.encode`

The same functions as `Factorizer.__call__`

<br>

#### `Factorizer.decode`

Takes the factorized indices and decodes them back to string (also accepts a batched input)

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `indices` (Union[List[Tuple[int, int, int]], List[List[Tuple[int, int, int]]]]) | the factorized indices |

**Returns**: Union[str, List[str]]


<br>


## Please cite the following publication (just arXiv for now)
```bibtex

```
