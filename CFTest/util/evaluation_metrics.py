#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2019/5/2 11:14
# @Author  : Cathy 
# @FileName: evaluation_metrics.py
import math
import random
import util.reader as reader



def ReadData():
    """
    获取data
    """
    data = dict()
    with open('../data/ratings.txt') as file_object:

        num = 0

        for line in file_object:
            if num == 0:
                num += 1
                continue

            item = line.strip().split(',')
            if len(item) < 4:
                continue
            [userid, itemid, rating, timestamp] = item

            user = userid
            item = itemid
            if user not in data.keys():
                data[user] = []
            data[user].append(item)
    return data


def SplitDataViaParity():
    filepath = '../data/ratings.txt'
    path1 = '../data/split/1.txt'
    path2 = '../data/split/2.txt'
    i = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            i = i+1
            if i % 2 != 0:
                g = open(path1, 'a')
                g.write(line)
                # print(line)
            else:
                q = open(path2, 'a')
                q.write(line)
                # print(line)


def SplitData(data, M, k, seed):
    """
    将用户行为数据集按照均匀分布随机分成M份，并挑选一份作为测试集
    :return:  train , test
    """

    # M = 8

    test = dict()
    train = dict()
    test_user = []
    train_user = []
    random.seed(seed)
    for user, items in data.items():
        for item in items:
            if random.randint(0, M) == k:
                if user not in test_user:
                    test_user.append(user)
                    test[user] = []
                    test[user].append(item)
            else:
                if user not in train_user:
                    train_user.append(user)
                    train[user] = []
                    train[user].append(item)
    return train, test


def GetRecommendation(user, rank, N):

    # rank 表示对指定User的推荐结果
    # rank =

    rank_N = dict()
    for i, pui in sorted(rank.items(), reverse=True)[0:N]:
        rank_N[i] = pui
    return rank_N


def Recall(train, test, N):
    """
    召回率
    """
    hit = 0
    all = 0

    for user in train.keys():
        tu = test[user]

        # rank 表示某一个用户的推荐结果
        rank = GetRecommendation(user, N)

        for item in rank:
            if item in tu :
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Precision(train, test, N):
    """
    准确率
    """
    hit = 0
    all = 0

    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)


def Coverage(train, test, N):
    """
    覆盖率，反映了土建算法发掘长尾的能力，覆盖率越高，说明推荐算法能够将长尾中的物品推荐给用户
    """
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)


def Populartity(train, test, rank, N):
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
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret


if __name__=="__main__":
    SplitDataViaParity()