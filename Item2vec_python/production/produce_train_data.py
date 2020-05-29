# -*- coding: utf-8 -*-
# @Time    : 2019/3/27 14:57
# @Author  : Cathy
# @FileName: produce_train_data.py
# @Software: PyCharm
import os
import sys

def produce_train_data(input_file,out_file):
    """
    获得训练数据，即从用户的行为序列中提取出对应的句子
    :param input_file:user behavior file
    :param out_file:output file
    """

    if not os.path.exists(input_file):
        return

    # 用于激励用户喜欢过的物品
    record = {}

    linenum = 0
    score_thr = 4.0

    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        # 按逗号分割
        item = line.strip().split(',')

        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])

        # 如果评分小于score_thr,则表示用户不喜欢该物品
        if rating < score_thr:
            continue
        if userid not in record:
            record[userid] = []
        record[userid].append(itemid)

    fp.close()

    # 创建输出文件
    fw = open(out_file,'w+')

    # 为产出较小文件而定义的计数器
    num = 0

    for userid in record:
        if num > 1000:
            break
        num += 1
        # fw.write(" ".join(record[userid]) + "\n")
        fw.write(" ".join(record[userid]))
    fw.close()


if __name__ == "__main__":

    produce_train_data("../data/originalData/ratings.txt", "../data/train_data_test.txt")
