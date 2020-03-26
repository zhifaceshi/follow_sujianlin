#%%
from allennlp.modules.matrix_attention import LinearMatrixAttention
from tqdm import tqdm
import random
from pprint import pprint
import os
import json
import collections
from typing import List, Dict, Tuple
import logging
import fire
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
from pathlib import Path
from allennlp.commands import main

override_dict = {"train_data_path":"/storage/gs2018/liangjiaxi/CORPUS/DATASET/china-people-daily-ner-corpus/fake",
                 "validation_data_path": "/storage/gs2018/liangjiaxi/CORPUS/DATASET/china-people-daily-ner-corpus/fake",
                 "test_data_path":"/storage/gs2018/liangjiaxi/CORPUS/DATASET/china-people-daily-ner-corpus/fake"
                 }
override_dict = json.dumps(override_dict).replace(' ', "").replace('\n', '')
def run(exp_name, config_file, ):
    print(os.getcwd())
    command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/debug/{exp_name}/{config_file.split('.')[0]}  -f  -o {override_dict} --include-package allennlp_plugins"
    print(sys.argv)
    print(command)
    sys.argv = command.split()

    main()

#%%
run('ner', 'bert1')
#%%
# run('ner', 'lstm')
