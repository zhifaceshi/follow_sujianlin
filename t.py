#%%
import itertools

def is_divide(line):
    if len(line.strip()) != 0:
        return False
    else:
        return True

with open('/storage/gs2018/liangjiaxi/CORPUS/DATASET/china-people-daily-ner-corpus/example.train') as f:
    for is_divider, lines in itertools.groupby(f, is_divide):
        print(is_divider)
        print(list(lines))

#%%
