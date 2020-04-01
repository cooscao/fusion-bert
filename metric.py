# -*- encoding: utf-8 -*-
'''
@File    :   metric.py
@Time    :   2020/04/01 14:09:38
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import numpy as np

def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


def mean_average_precision(y_true: np.array, y_pred:  np.array, threshold: float=0.) ->float:
    result = 0.
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, _) in enumerate(coupled_pair):
        if label > threshold:
            pos += 1.
            result += pos / (idx + 1.)
    if pos == 0:
        return 0.
    else:
        return result / pos


def mean_reciprocal_rank(y_true: np.array, y_pred: np.array, threshold: float=0.) -> float:
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, _) in enumerate(coupled_pair):
        if label > threshold:
            return 1. / (idx + 1)
    return 0.


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)