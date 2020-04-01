# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/03/31 23:55:51
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2019-2020, MILAB_SCU
@Desc    :   bert model
'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class FusionBertModule(nn.Module):
    """
    when config.output_hidden_states=True,return last hidden output
    (batch_size, sequence_length, hidden_size)
    """
    def __init__(self, config):
        super(FusionBertModule, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.pooler_mode = config.pooler_mode
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, sep_idx=-1):
        outputs = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        sequence_output, pooled_output = outputs[:2]
        ## 单文档
        if sep_idx == -1:
            # mean pooling
            input_mask_expanded = input_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            mean_pooling_out = sum_embeddings / sum_mask  # [batch_size, hidden_size]
        else:
            mean_pooling_out = self.dropout(pooled_output) #[batch_size, hidden_size]
        return mean_pooling_out
            

class FusionBert(nn.Module):
    def __init__(self, config):
        super(FusionBert, self).__init__()
        self.num_labels = 2
        self.bert_module = FusionBertModule(config)
        self.linear1 = nn.Linear(config.hidden_size, self.num_labels)
        self.linear2 = nn.Linear(config.hidden_size * 4, config.hidden_size)

    def forward(self, x_input_ids, x_segment_ids, x_input_mask,
                      y_input_ids, y_segment_ids, y_input_mask,
                      xy_input_ids, xy_segment_ids, xy_input_mask, labels):
        x_output = self.bert_module(x_input_ids, x_segment_ids, x_input_mask)
        y_output = self.bert_module(y_input_ids, y_segment_ids, y_input_mask)
        abs_output = torch.abs(x_output - y_output)
        simanse_embedding = torch.cat((x_output, y_output, abs_output), 1)
        
        xy_output = self.bert_module(xy_input_ids, xy_segment_ids, xy_input_mask, 0)
        output = torch.cat((xy_output, simanse_embedding), 1) #[batch_size, hidden_size * 4]
        output = F.relu(self.linear2(output))
        logits = self.linear1(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



