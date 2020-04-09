# -*- encoding: utf-8 -*-
'''
@File    :   embed.py
@Time    :   2020/04/09 17:05:35
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2019-2020, MILAB_SCU
@Desc    :   None
'''

import csv
import numpy as np
import logging
import os
import fire
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert(csv_file, model, ignore_header=False):
    pairs = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for idx, line in enumerate(reader):
            if ignore_header and idx == 0: continue
            embedding = model.encode(line[:-1])
            pairs.append(embedding)
    array = np.vstack((pairs[0][0], pairs[0][1]))
    for pair in tqdm(pairs[1:], desc="convert to one array"):
        array = np.vstack((array, pair[0]))
        array = np.vstack((array, pair[1]))
    return array

def main(data_dir, train=True, dev=True, test=True, model_name='bert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    if train:
        logger.info("Start convert train.tsv")
        train_csv = os.path.join(data_dir, 'train.tsv')
        assert os.path.exists(train_csv), "File path not exists."
        train_array = convert(train_csv, model)
        train_npy = os.path.join(data_dir, "train_sentence.npy")
        np.save(train_npy, train_array)
    if dev:
        logger.info("Start convert dev.tsv")
        dev_csv = os.path.join(data_dir, 'dev.tsv')
        assert os.path.exists(dev_csv), "File path not exists."
        dev_array = convert(dev_csv, model)
        dev_npy = os.path.join(data_dir, "dev_sentence.npy")
        np.save(dev_npy, dev_array)
    if test:
        logger.info("Start convert test.tsv")
        test_csv = os.path.join(data_dir, "test.tsv")
        assert os.path.exists(test_csv), "File path not exists."
        test_array = convert(test_csv, model)
        test_npy = os.path.join(data_dir, "test_sentence.npy")
        np.save(test_npy, test_array)
    logger.info("Successfully converted npy.")

if __name__ == "__main__":
    fire.Fire(main)
    
        
    