from factorizer import Factorizer


tokenizer = Factorizer("english.dawg")
sentence = "The echo of a distant time comes willowing across the sand, and everything is green and submarine."

encoding = tokenizer(sentence)

print(f"INPUT:    {sentence}")
print(f"SUBWORDS: {' '.join(encoding.tokens)}")
print(f"INDICES:  {' '.join(str(index) for index in encoding.ids)}")
print(f"DECODED:  {tokenizer.decode(encoding.ids, skip_special_tokens=False)}")
