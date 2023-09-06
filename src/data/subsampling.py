import pandas as pd
from open_clip.tokenizer import SimpleTokenizer
import collections
import numpy as np
import json

df_cc3m = pd.read_csv('../path/to/cc3m_train.csv', sep='\t')
tokenizer = SimpleTokenizer(subsample_file=None)
# build vocab
words = []
for caption in df_cc3m["caption"].tolist():
    text = tokenizer.encode_text(caption)
    words.append(text)

counter = collections.Counter([tk for st in words for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
total_count = sum(counter.values())
counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
print("Total words:", total_count)

threshold = 1e-4
freqs = {word: count / total_count for word, count in counter.items()}  # f(w_i)
p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in counter}  # P(w_i)
# Save the p_drop to a json file
with open('../open_clip/model_configs/cc3m_train_p_sub_1e4.json', 'w', encoding='utf-8') as f:
    json.dump(p_drop, f)
