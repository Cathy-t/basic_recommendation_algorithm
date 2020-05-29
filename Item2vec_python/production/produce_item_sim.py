# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 15:53
# @Author  : Cathy
# @FileName: produce_item_sim.py
# @Software: PyCharm
import operator

import numpy as np
import os


def load_item_vec(input_file):
    """
    将产生的item_vec加载到内存中
    :param input_file:item vec file
    :return:dict key:itemid value:(每个Item对应的向量列表)np.array[num1,num2,.....]
    """
    if not os.path.exists(input_file):
        print("there is not a txt........")
        return {}
    linenum = 0;

    # 定义一个数据结构，用于记录返回结果
    item_vec = {}

    fp = open(input_file)
    for line in fp:

        item = line.strip().replace('[', '').replace(']', '').split()  # 去除一行中的空格、以及[]

        itemid = item[0]
        if itemid == "</s>":
            continue

        # 使用一个列表推导式
        item_vec[itemid] = np.array([float(ele) for ele in item[1:]])

    fp.close()
    return item_vec


def cal_item_sim(item_vec, itemid, output_file):
    """
    :param item_vec: item embedding vector
    :param itemid: fixed itemid to clac item sim
    :param output_file: the file to store result
    """

    if itemid not in item_vec:
        print("itemid not in item_vec.......")
        return
    score = {}
    topk = 10
    fix_item_vec = item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid == itemid:
            continue
        tmp_itemvec = item_vec[tmp_itemid]
        fenmu = np.linalg.norm(fix_item_vec) * np.linalg.norm(tmp_itemvec)
        if fenmu == 0:
            score[tmp_itemid] = 0
        else:
            score[tmp_itemid] = round(np.dot(fix_item_vec, tmp_itemvec) / fenmu, 3)

    fw = open(output_file, 'w+')
    out_str = itemid + "\t"
    tmp_list = []
    for zuhe in sorted(score.items(), key=operator.itemgetter(1), reverse=True)[:topk]:
        tmp_list.append(zuhe[0] + "_" + str(zuhe[1]))
    out_str += ";".join(tmp_list)
    print(out_str + "\n")
    fw.write(out_str + "\n")
    fw.close()


def run_main(input_file, output_file):
    item_vec = load_item_vec(input_file)
    cal_item_sim(item_vec, "101", output_file)


if __name__ == "__main__":

    run_main("../production/log_result_embedding_test/item_vec.txt", "../data/sim_result101.txt")





