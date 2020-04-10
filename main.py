# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/04/01 14:14:38
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pprint import pprint
from model import FusionBert, FusionSentenceBert
from metric import mean_average_precision, mean_reciprocal_rank, accuracy
from util import InputExample, InputFeatures, TrecProcessor, MrpcProcessor, QqpProcessor,convert_examples_to_features, get_datasets
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import f1_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    processors = {
        "trec": TrecProcessor,
        "mrpc": MrpcProcessor,
        "qqp": QqpProcessor
    }

    num_labels_task = {
        "trec": 2,
        "mrpc": 2,
        "qqp": 2
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
    model = FusionSentenceBert(args.bert_model, config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.do_train:
        train(model, processor,optimizer, train_examples, label_list, args, tokenizer,
              device, n_gpu, num_train_optimization_steps, valid=True)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # eval_dataloader = get_dataloader(processor, args, tokenizer, 'test')
        # eval(model, eval_dataloader, device)
        test_file = os.path.join(args.data_dir, 'test.tsv')
        map_eval(test_file, args.max_seq_length, tokenizer, device, model, label_list)
    # save model
        # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    # config = BertConfig(output_config_file)
    # model = FusionBert(config=config)
    # model.load_state_dict(torch.load(output_model_file))
    # # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    # model.to(device)


def train(model, processor, optimizer, train_examples, label_list, args, tokenizer, device, n_gpu, num_train_optimization_steps,valid=True):
    # model.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features], dtype=torch.long)
    # x_input_ids = torch.tensor(
    #     [f.input_ids_x for f in train_features], dtype=torch.long)
    # x_input_mask = torch.tensor(
    #     [f.input_mask_x for f in train_features], dtype=torch.long)
    # x_segment_ids = torch.tensor(
    #     [f.segment_ids_x for f in train_features], dtype=torch.long)
    # y_input_ids = torch.tensor(
    #     [f.input_ids_y for f in train_features], dtype=torch.long)
    # y_input_mask = torch.tensor(
    #     [f.input_mask_y for f in train_features], dtype=torch.long)
    # y_segment_ids = torch.tensor(
    #     [f.segment_ids_y for f in train_features], dtype=torch.long)
    # embedding
    embedding_xs = torch.tensor(
        [f.embeddimng_x.tolist() for f in train_features], dtype=torch.float)
    embedding_ys = torch.tensor(
        [f.embeddimng_y.tolist() for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                               embedding_xs, embedding_ys)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, embedding_x, embedding_y = batch
            loss = model(input_ids, segment_ids, input_mask, embedding_x, embedding_y, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            logger.info(loss.item())

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * \
                        warmup_linear(
                            global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                #global_step += 1
        if valid:
            logging.info('Start eval the dev set')
            # eval_dataloader = get_dataloader(processor,args, tokenizer,mode='dev')
            # eval(model, eval_dataloader, device)
            dev_file = os.path.join(args.data_dir, 'dev.tsv')
            map_eval(dev_file, args.max_seq_length, tokenizer, device, model, label_list)

def eval(model, eval_dataloader, device):
    model.eval()
    eval_accuracy = 0.
    #eval_map, eval_accuracy, eval_mrr = 0., 0., 0.
    nb_eval_steps, nb_eval_examples = 0, 0
    preds, labels = [],[]

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = (t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, x_input_ids, x_input_mask, x_segment_ids, y_input_ids, y_input_mask, y_segment_ids = batch

        with torch.no_grad():
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(x_input_ids, x_input_mask, x_segment_ids,
                            y_input_ids, y_input_mask, y_segment_ids,
                            input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        preds.extend(np.argmax(logits, 1).tolist())
        labels.extend(label_ids.tolist())
        tmp_eval_accuracy = accuracy(logits, label_ids)
        #tmp_eval_map = mean_average_precision(label_ids, logits[:,1])
        #tmp_eval_mrr = mean_reciprocal_rank(label_ids, logits[:, 1])

        # eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        #eval_map += tmp_eval_map
        #eval_mrr += tmp_eval_mrr

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    # eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    eval_f1 = f1_score(np.array(labels), np.array(preds)) 
    #eval_map = eval_map / nb_eval_examples
    #eval_mrr = eval_mrr / nb_eval_examples
    result = {  # 'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'eval_f1_score': eval_f1}
        #'eval_map': eval_map,
        #'eval_mrr': eval_mrr}
    pprint(result)

    

def get_dataloader(processor, args, tokenizer, mode='test'):
    eval_examples = processor.get_test_examples(args.data_dir) if mode=='test' \
        else processor.get_dev_examples(args.data_dir)
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
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
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    return eval_dataloader



def map_eval(eval_file, token_length, tokenizer, device, model, label_list):
    model.eval()
    datasets, labels = get_datasets(eval_file) 
    total_batches = 0
    total_avp = 0.0
    total_mrr = 0.0
    for k, dataset in tqdm(datasets.items(), desc="Eval datasets"):
        examples = []
        for i, data in enumerate(dataset):
            examples.append(InputExample(i, data[0], data[1], '0'))
        eval_features = convert_examples_to_features(examples, label_list,
                                                    token_length, tokenizer)
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long).to(device)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long).to(device)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long).to(device)
        # all_label_ids = torch.tensor(
        #   [f.label_id for f in eval_features], dtype=torch.long).to(device)
        embedding_x = torch.tensor(
            [f.embeddimng_x.tolist() for f in eval_features], dtype=torch.float).to(device)
        embedding_y = torch.tensor(
            [f.embeddimng_y.tolist() for f in eval_features], dtype=torch.float).to(device)
        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask, embedding_x, embedding_y)
        score = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        label = np.array(list(map(int, labels[k])))
        # print(score, label)
        total_avp += mean_average_precision(label, score)
        total_mrr += mean_reciprocal_rank(label, score)
        total_batches += 1
    mAP = total_avp / total_batches
    mRR = total_mrr / total_batches
    logger.info("map is : {}, mrr is : {}".format(mAP, mRR))


if __name__ == "__main__":
    main()
