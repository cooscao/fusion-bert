# -*- encoding: utf-8 -*-
'''
@File    :   test_cmed.py
@Time    :   2019/12/03 14:24:45
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   测试模型在测试集上的效果
'''

from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import random

import fire
import numpy as np
import pandas as pd
import torch
from model import FusionBert
from util import InputExample, InputFeatures, convert_examples_to_features, get_datasets
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForSequenceClassification)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

# from run_cmed import InputExample, InputFeatures, _truncate_seq_pair

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCmedProcessor(object):
    def __init__(self, dataset_path, cache='./test_cache/'):
        self.question_path = os.path.join(dataset_path, 'question.csv')
        self.answer_path = os.path.join(dataset_path, 'answer.csv')
        self.cache = cache
        if not os.path.exists(self.cache):
            os.makedirs(self.cache)
        logger.info("Reading the df from train and dev csv....")
        self.question_df = pd.read_csv(
            self.question_path, sep=',', encoding='utf-8')
        self.answer_df = pd.read_csv(
            self.answer_path, sep=',', encoding='utf-8')
        self.test_path = os.path.join(dataset_path, 'test_candidates.txt')

    def get_test_example(self):
        cache_path = os.path.join(self.cache, "test.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading test examples from cache..")
            with open(cache_path, 'rb') as fr:
                test_examples = pickle.load(fr)
        else:
            logger.info("Loading test examples from file..")
            test_examples = self._create_examples(self.test_path)
            logger.info("Saving test examples to cache...")
            with open(cache_path, 'wb') as fw:
                pickle.dump(test_examples, fw)
        return test_examples

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, candidate_path):
        examples = []
        df = pd.read_csv(candidate_path, sep=',')
        group_df = df.groupby('question_id')
        for group_id, group_data in tqdm(group_df):
            group_examples = []
            question = self.question_df[self.question_df['question_id']
                                        == group_id]['content'].values[0]
            #pos_ids = group_data['pos_ans_id'].drop_duplicates().values
            #neg_ids = random.sample(group_data['neg_ans_id'].values.tolist(), self.neg_num)
            ans_ids = group_data['ans_id'].values
            labels = group_data['label'].values
            # 加入正样本
            for ans_id, label in zip(ans_ids, labels):
                guid = "%s-%s" % (str(group_id), str(ans_id))
                text_a = question
                text_b = self.answer_df[self.answer_df['ans_id']
                                        == ans_id]['content'].values[0]
                label = str(label)
                group_examples.append(
                    InputExample(guid=guid, text_a=text_a,
                                 text_b=text_b, label=label)
                )
            # 打乱顺序并记录正样本位置
            random.shuffle(group_examples)
            pos_idxs = []
            for idx, example in enumerate(group_examples):
                if example.label == "1":
                    pos_idxs.append(idx)
            examples.append({'examples': group_examples, 'pos_ids': pos_idxs})
        return examples
    
    
def get_dataloader(processor, tokenizer, mode='test'):
    eval_examples = processor.get_test_examples() if mode=='test' \
        else processor.get_dev_examples()
    eval_examples = eval_examples[:1000]
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features(
        eval_examples, label_list, 256, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", 8)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)
    x_input_ids = torch.tensor(
        [f.input_ids_x for f in eval_features], dtype=torch.long)
    x_input_mask = torch.tensor(
        [f.input_mask_x for f in eval_features], dtype=torch.long)
    x_segment_ids = torch.tensor(
        [f.segment_ids_x for f in eval_features], dtype=torch.long)
    y_input_ids = torch.tensor(
        [f.input_ids_y for f in eval_features], dtype=torch.long)
    y_input_mask = torch.tensor(
        [f.input_mask_y for f in eval_features], dtype=torch.long)
    y_segment_ids = torch.tensor(
        [f.segment_ids_y for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                x_input_ids, x_input_mask, x_segment_ids,
                                y_input_ids, y_input_mask, y_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=8)
    return eval_dataloader
    
def test_fusion(data_dir, model_path, bert_path, mode='point', max_seq_length=256):
    assert mode in ['point', 'pair', 'ensemble'], "mode must be point or pair or ensemble"
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # n_gpu = torch.cuda.device_count()
    processor = TestCmedProcessor(data_dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # 加载模型
    logger.info("Loading model from trained dir....")
    output_config_file = os.path.join(model_path, 'bert_config.json')
    # model = FusionBert(args.bert_model, config)
    config = BertConfig(output_config_file)
    model = FusionBert(bert_path, config)
    # model = BertForSequenceClassification(config, num_labels=num_labels if mode!='pair' else 1)
    output_model_file = os.path.join(model_path, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    model.eval()
    logger.info("Loading model Done")
    
    eval_dataloader = get_dataloader(processor, tokenizer)

    
    predicts, corrects = 0, 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = (t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, x_input_ids, x_input_mask, x_segment_ids, y_input_ids, y_input_mask, y_segment_ids = batch

        with torch.no_grad():
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(x_input_ids, x_input_mask, x_segment_ids,
                            y_input_ids, y_input_mask, y_segment_ids,
                            input_ids, segment_ids, input_mask)[:1]
        logits = logits.detach().cpu().numpy()

        predict_num = len(pos_ids)
        argidxs = logits.argsort()[::-1][:predict_num].tolist()
        corrects += len(set(argidxs) & set(pos_ids))
        predicts += len(argidxs)
    logger.info("Eval result = {}".format(corrects / predicts))



if __name__ == "__main__":
    fire.Fire(test_fusion)
