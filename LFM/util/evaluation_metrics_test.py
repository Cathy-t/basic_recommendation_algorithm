#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2019/5/10 21:52
# @Author  : Cathy 
# @FileName: evaluation_metrics_test.py
import math
import random


def SplitDataViaParity():
    filepath = '../data/ratings.txt'
    path1 = '../data/split/rat_train.txt'
    path2 = '../data/split/rat_test.txt'

    # i_aim = random.randint(0, 9)
    i = 0
    num = 0

    with open(filepath, 'r', errors='ignore') as f:
        for line in f.readlines():
            if num == 0:
                num += 1
                continue
            else:
                i += 1
                if i % 10 == 1 | 3 | 5 | 7:
                    g = open(path2, 'a')
                    g.write(line)
                    # print(line)
                else:
                    q = open(path1, 'a')
                    q.write(line)


def Recall(train, test):
    """
    召回率
    """
    hit = 0
    all = 0

    for user in train.keys():

        if user not in test.keys():
            continue
        else:
            tu = test[user]

        # rank 表示某一个用户的推荐结果
        rank = train[user]

        for item in rank:
            if item in tu:
                hit += 1

        all += len(tu)

    return hit / (all * 1.0)


def Precision(train, test):
    """
    准确率
    """
    hit = 0
    all = 0

    for user in train.keys():

        if user not in test.keys():
            continue
        else:
            tu = test[user]

        # rank 表示某一个用户的推荐结果
        rank = train[user]

        for item in rank:
            if item in tu:
                hit += 1
        all += len(rank)

    return hit / (all * 1.0)


def Coverage(train, item_info):
    """
    覆盖率，反映了推荐算法发掘长尾的能力，覆盖率越高，说明推荐算法能够将长尾中的物品推荐给用户
    """
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in item_info:
            all_items.add(item)
        rank = train[user]
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)


def Populartity(train):
    """
    新颖度，用推荐列表中物品的平均流行度度量推荐结果的新颖度。
    若推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖
    """
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = train[user]
        for item in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret


if __name__ == "__main__":
    SplitDataViaParity()
