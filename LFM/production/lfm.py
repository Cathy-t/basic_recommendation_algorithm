# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 10:32
# @Author  : Cathy
# @FileName: lfm.py
# @Software: PyCharm


"""
此为 lfm 函数训练的代码
"""
import operator

import numpy as np

import sys
sys.path.append("../util")
import util.read as read

def lfm_train(train_data,F,alpha,beta,step):
    """
    :param train_data: 训练样本 train_data for lfm
    :param F: 维度 user vector len,item vector len
    :param alpha: 正则化参数 regularization factor
    :param beta: 模型的学习率 learning rate
    :param step: 模型迭代的次数 iteration num
    :return:  dict: key itemid,value:np.ndarray       dict: key userid,value:np.ndarray
    """

    user_vec = {}
    item_vec = {}

    for step_index in range(step):
        for data_instance in train_data:
            userid,itemid,label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
        delta = label - model_predict(user_vec[userid], item_vec[itemid])
        for index in range(F):
            user_vec[userid][index] += beta *(delta*item_vec[itemid][index] - alpha*user_vec[userid][index])
            item_vec[itemid][index] += beta *(delta*user_vec[userid][index] - alpha*item_vec[itemid][index])
        # 将学习率进行一个衰减
        beta = beta * 0.9
    return user_vec,item_vec

def init_model(vector_len):
    """
    初始化函数
    :param vector_len: the len of vextor
    :return: a ndarray
    """

    # 使用标准的正态分布来初始化
    return np.random.randn(vector_len)

def model_predict(user_vector,item_vector):
    """
    user_vector and item_vector distance
    :param user_vector: model produce user vector
    :param item_vector: model produce item vector
    :return: a num: user vector 和 item vector 距离的远近
    """

    res = np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res

def model_train_process():
    """
    test lfm model train
    """
    train_data = read.get_train_data("../data/split/rat_train.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)
    # print(user_vec["1"])
    # print(item_vec["50"])

    # 基于一个用户的推荐
    # recom_result = give_recom_result(user_vec,item_vec,'1')
    # ana_recom_result(train_data,'1',recom_result)

    # 基于所有用户的推荐
    for userid in user_vec:
        recom_result = give_recom_result(user_vec,item_vec,userid)

        # 在此处定义一个数据结构(字典)中存储【userid,[itemid1,itemid2]】


        ana_recom_result(train_data, userid, recom_result)


def give_recom_result(user_vec, item_vec, userid):
    """
    user lfm model result give fix userid recom result
    :param user_vec: lfm model result
    :param item_vec: lfm model result
    :param userid: user
    :return: list :[(itemid,score),(itemid1,score1)]
    """

    fix_num = 80

    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec:
        item_vector = item_vec[itemid]
        # 使用欧式距离来计算
        res = np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
        record[itemid] = res
    for zuhe in sorted(record.items(),key=operator.itemgetter(1),reverse=True)[:fix_num]:
        itemid = zuhe[0]
        score = round(zuhe[1], 3)

        recom_list.append((itemid,score))
    return recom_list

# debug 函数
def ana_recom_result(train_data,userid,recom_list):
    """
    debug recom result for userid 此函数用来评价推荐结果的好坏
    :param train_data: 训练数据，我们想展示一下之前用户对哪些Item感兴趣
    :param userid: 分析哪个用户的推荐结果
    :param recom_list: 模型给出的推荐结果
    """
    item_info = read.get_item_info("../data/movies.txt")
    for data_instance in train_data:
        tmp_userid, itemid, label = data_instance
        if tmp_userid == userid and label == 1:
            print(item_info[itemid])
    print("recom_result")
    for zuhe in recom_list:
        print(item_info[zuhe[0]])


if __name__ =="__main__":
    model_train_process()