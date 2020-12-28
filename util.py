# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2020/04/01 00:46:09
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import csv
import os
import sys
import logging
import pickle
import pandas as pd
import random
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                       input_ids_x, input_mask_x, segment_ids_x,
                       input_ids_y, input_mask_y, segment_ids_y):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_ids_x = input_ids_x
        self.input_mask_x = input_mask_x
        self.segment_ids_x = segment_ids_x
        self.input_ids_y = input_ids_y
        self.input_mask_y = input_mask_y
        self.segment_ids_y = segment_ids_y


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                #if sys.version_info[0] == 2:
                #    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class TrecProcessor(DataProcessor):
    """Processor for the TREC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class NlpccProcessor(DataProcessor):
    """Processor for the TREC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class LcqmcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or len(line) != 6: continue
            guid = "%s-%s" % (set_type, i)
            #print(line)
            text_a = line[3]
            text_b = line[4]
            label = line[5]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
class CmedQAProcess(object):
    def __init__(self, dataset_path, cache='./cache_cmed/', neg_num=10):
        # self.dataset_path = dataset_path
        self.question_path = os.path.join(dataset_path, 'question.csv')
        self.answer_path = os.path.join(dataset_path, 'answer.csv')
        logger.info("Reading the df from train and dev csv....")
        self.question_df = pd.read_csv(self.question_path, sep=',', encoding='utf-8')
        self.answer_df = pd.read_csv(self.answer_path, sep=',', encoding='utf-8')
        self.train_path = os.path.join(dataset_path, 'train_candidates.txt')
        self.dev_path = os.path.join(dataset_path, 'dev_candidates.txt')
        self.test_path = os.path.join(dataset_path, 'test_candidates.txt')
        self.neg_num = neg_num
        self.cache = cache
        # self.cache = os.path.join(dataset_path, 'cache_cmed')
        if not os.path.exists(self.cache):
            os.makedirs(self.cache)
        

    def get_train_examples(self):
        cache_path = os.path.join(self.cache, "train.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading train examples from cache..")
            with open(cache_path, 'rb') as fr:
                train_examples = pickle.load(fr)
        else:
            logger.info("Loading train examples from file..")
            train_examples = self._create_examples(self.train_path)
            logger.info("Saving train examples to cache...")
            with open(cache_path, 'wb') as fw:
                pickle.dump(train_examples, fw)
        return train_examples

    def get_dev_examples(self):
        cache_path = os.path.join(self.cache, "dev.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading dev examples from cache..")
            with open(cache_path, 'rb') as fr:
                dev_examples = pickle.load(fr)
        else:
            logger.info("Loading dev examples from file..")
            dev_examples = self._create_dev_examples(self.dev_path)
            logger.info("Saving dev examples to cache...")
            with open(cache_path, 'wb') as fw:
                pickle.dump(dev_examples, fw)
        return dev_examples
    
    def get_test_examples(self):
        cache_path = os.path.join(self.cache, "dev.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading dev examples from cache..")
            with open(cache_path, 'rb') as fr:
                dev_examples = pickle.load(fr)
        else:
            logger.info("Loading dev examples from file..")
            dev_examples = self._create_dev_examples(self.test_path)
            logger.info("Saving dev examples to cache...")
            with open(cache_path, 'wb') as fw:
                pickle.dump(dev_examples, fw)
        return dev_examples

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, candidate_path):
        logger.info('loading examples from {}'.format(candidate_path))
        examples = []
        df = pd.read_csv(candidate_path, sep=',')
        group_df = df.groupby('question_id')
        group_num = 0
        for group_id, group_data in tqdm(group_df):
            question = self.question_df[self.question_df['question_id'] == group_id]['content'].values[0]
            pos_ids = group_data['pos_ans_id'].drop_duplicates().values
            neg_ids = random.sample(group_data['neg_ans_id'].values.tolist(), self.neg_num)
            # 加入正样本
            for pos_id in pos_ids:
                guid = "%s-%s" % (str(group_id), str(pos_id))
                text_a = question
                text_b = self.answer_df[self.answer_df['ans_id'] == pos_id]['content'].values[0]
                label = "1"
                examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
            # 加入负样本
            for neg_id in neg_ids:
                guid = "%s-%s" % (str(group_id), str(neg_id))
                text_a = question
                text_b = self.answer_df[self.answer_df['ans_id'] == neg_id]['content'].values[0]
                label = "0"
                examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
            group_num += 1
            if group_num >= 20000:
                break
        return examples
            # 加入负样本

    def _create_dev_examples(self, candidate_path):
        logger.info('loading examples from {}'.format(candidate_path))
        examples = []
        df = pd.read_csv(candidate_path, sep=',')
        group_df = df.groupby('question_id')
        # group_num = 0
        for group_id, group_data in tqdm(group_df):
            question = self.question_df[self.question_df['question_id'] == group_id]['content'].values[0]
            #pos_ids = group_data['pos_ans_id'].drop_duplicates().values
            #neg_ids = random.sample(group_data['neg_ans_id'].values.tolist(), self.neg_num)
            ans_ids = group_data['ans_id'].values
            labels = group_data['label'].values
            # 加入正样本
            for ans_id, label in zip(ans_ids, labels):
                guid = "%s-%s" % (str(group_id), str(ans_id))
                text_a = question
                text_b = self.answer_df[self.answer_df['ans_id'] == ans_id]['content'].values[0]
                label = str(label)
                examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (_, example) in tqdm(enumerate(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        if not example.text_b: continue

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # single of x
        tokens_x = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_x = [0] * len(tokens)
        tokens_y = ["[CLS]"] + tokens_b + ["[SEP]"]
        segment_ids_y = [0] * len(tokens_y)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_x = tokenizer.convert_tokens_to_ids(tokens_x)
        input_ids_y = tokenizer.convert_tokens_to_ids(tokens_y)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        input_mask_x = [1] * len(input_ids_x)
        input_mask_y = [1] * len(input_ids_y)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding 
        
        padding_x = [0] * (max_seq_length - len(input_ids_x))
        input_ids_x += padding_x
        input_mask_x += padding_x
        segment_ids_x += padding_x

        padding_y = [0] * (max_seq_length - len(input_ids_y))
        input_ids_y += padding_y
        input_mask_y += padding_y
        segment_ids_y += padding_y

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ids_x) == max_seq_length
        assert len(input_mask_x) == max_seq_length
        assert len(segment_ids_x) == max_seq_length
        assert len(input_ids_y) == max_seq_length
        assert len(input_mask_y) == max_seq_length
        assert len(segment_ids_y) == max_seq_length

        label_id = label_map[example.label]
        

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              input_ids_x=input_ids_x,
                              input_mask_x=input_mask_x,
                              segment_ids_x=segment_ids_x,
                              input_ids_y=input_ids_y,
                              input_mask_y=input_mask_y,
                              segment_ids_y=segment_ids_y))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def get_datasets(filepath, skip_header=True):
    datasets = []
    d = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter='\t')
        for idx, line in enumerate(reader):
            if skip_header and idx == 0: continue
            d[line[0]].append(line[1:])
    labels = {}
    datasets = {}
    for k, v in d.items():
        # k is the id, v is the dataset
        dataset = []
        label = []
        for _, pair in enumerate(v):
            dataset.append([k, pair[0]])
            label.append(pair[-1])
        datasets[k] = dataset
        labels[k] = label
    logging.info("Done with loading eval datasets")
    return datasets, labels
