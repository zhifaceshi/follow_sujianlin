"""
@Time    :2020/2/12 21:04
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""

import json
import os
from typing import List, Dict, Tuple
import logging
import fire
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
from pathlib import Path
from allennlp.commands import main
override_dict = {
                 }
override_dict = json.dumps(override_dict).replace(' ', "").replace('\n', '')
def run(exp_name, config_file, ):
    print(os.getcwd())
    command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/{exp_name}/{config_file.split('.')[0]}  -f  -o {override_dict}    --include-package allennlp_plugins"
    print(sys.argv)
    print(command)
    sys.argv = command.split()

    main()
if __name__ == '__main__':

    fire.Fire(run)
