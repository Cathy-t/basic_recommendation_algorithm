# -*- coding: utf-8 -*-
# @Time    : 2019/2/24 20:40
# @Author  : Cathy
# @FileName: itemcf.py
# @Software: PyCharm

from __future__ import division
import sys
sys.path.append("../util")
import util.reader as reader
import math
import operator

import util.evaluation_metrics_test as evaluation

def base_contribute_score():
    """
     item cf base sim contribution score by user
    """
    return 1

# 【升级版1：活跃用户应该被降低在相似度】
def update_one_contribute_score(user_total_click_num):
    """
    itemcf update sim contribution score by user
    """
    return 1/math.log10(1 + user_total_click_num)

# 【升级版2：用户在不同时间对item的操作应给予时间衰减惩罚】
def update_two_contribute_score(click_time_one,click_time_two):
    """
    item cf update two sim contribution score by score
    """
    delata_time = abs(click_time_one - click_time_two)

    # 将时间戳的秒，变成天
    total_sec = 60 * 60 * 24
    delata_time = delata_time/total_sec

    return 1/(1 + delata_time)

# 功能函数，其中user_click_time为公式升级2 所需要
# 【升级2】寄托于同一个用户对不同item点击时间得差，差异值越小对相似度得贡献值越大
def cal_item_sim(user_click,user_click_time):
    """
    :param user_click: dict,key userid value[itemid1,itemid2]
    :return:dict,key:itemid_i,value dict,value_key itemid_j,value_value simscore
    """

    # 创建一个数据结构，用来存储有多少个公共用户分别点击了这两个Item
    co_appear = {}
    item_user_click_time= {}
    for user,itemlist in user_click.items():
        for index_i in range(0,len(itemlist)):
            itemid_i = itemlist[index_i]
            item_user_click_time.setdefault(itemid_i,0)
            item_user_click_time[itemid_i] += 1

            for index_j in range(index_i + 1,len(itemlist)):
                itemid_j = itemlist[index_j]

                if user + "_" + itemid_i not in user_click_time:
                    click_time_one = 0
                else:
                    click_time_one = user_click_time[user + "_" + itemid_i]
                if user + "_" + itemid_j not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user + "_" + itemid_j]

                # itemid_i对itemid_j的贡献
                co_appear.setdefault(itemid_i,{})
                co_appear[itemid_i].setdefault(itemid_j,0)

                # 【基础版】base_contribute_score此为基础贡献分
                # co_appear[itemid_i][itemid_j] += base_contribute_score()

                # 【在公式升级1部分】需要对用户的点击的item的数量做一定的惩罚，命名为updata_one_contribute_score()
                # itemlist表示该用户的总点击数量
                # co_appear[itemid_i][itemid_j] += update_one_contribute_score(len(itemlist))

                # 【公式升级2】，参数为两个点击事件
                co_appear[itemid_i][itemid_j] += update_two_contribute_score(click_time_one,click_time_two)

                # itemid_j对itemid_i的贡献
                co_appear.setdefault(itemid_j, {})
                co_appear[itemid_j].setdefault(itemid_i,0)
                # co_appear[itemid_j][itemid_i] += base_contribute_score()
                # co_appear[itemid_j][itemid_i] += update_one_contribute_score(len(itemlist))
                co_appear[itemid_j][itemid_i] += update_two_contribute_score(click_time_one,click_time_two)

    # item_sim_score的计算
    item_sim_score = {}
    item_sim_score_sorted = {}
    for itemid_i,relate_item in co_appear.items():
        for itemid_j,co_time in relate_item.items():
            sim_score = co_time/math.sqrt(item_user_click_time[itemid_i]*item_user_click_time[itemid_j])
            # 二维数据初始化的赋值
            item_sim_score.setdefault(itemid_i,{})
            item_sim_score[itemid_i].setdefault(itemid_j,0)
            item_sim_score[itemid_i][itemid_j] = sim_score
    for itemid in item_sim_score:
        item_sim_score_sorted[itemid] = sorted(item_sim_score[itemid].items(),key =\
            operator.itemgetter(1),reverse=True)

    return item_sim_score_sorted


def cal_recom_result(sim_info,user_click):
    """
    recom by itemcf
    :param sim_info:item sim dict
    :param user_click:user click dict
    :return:dict,key:userid value dict,value_key itemid,value_value recom_score   {userid,{itemid,recon_score}}
    """
    recent_click_num = 3
    topk = 80

    # 定义一个数据结构来存储推荐的结果
    recom_info = {}
    for user in user_click:
        click_list = user_click[user]
        recom_info.setdefault(user,{})
        for itemid in click_list[:recent_click_num]:
            if itemid not in sim_info:
                continue
            for itemsimzuhe in sim_info[itemid][:topk]:
                itemsimid = itemsimzuhe[0]
                itemsimscore = itemsimzuhe[1]
                recom_info[user][itemsimid] = itemsimscore
    return recom_info

# debug函数：item与item之间的相似度
def debug_itemsim(item_info,sim_info):
    """
    show itemsim info
    :param item_info:dict,key itemid value:[title,genres]
    :param sim_info:dict key itemid,value dict,key[(itemid1,simscore),(itemid2,simscore)]
    """
    # 指定一个debug的itemid
    fixed_itemid = "1"
    if fixed_itemid not in item_info:
        print("invalid itemid")
        return
    [title_fix,genres_fix] = item_info[fixed_itemid]
    for zuhe in sim_info[fixed_itemid][:5]:
        itemid_sim = zuhe[0]
        sim_score = zuhe[1]
        if itemid_sim not in item_info:
            continue
        [title,genres] = item_info[itemid_sim]
        # print(title_fix + "\t" + genres_fix + "\tsim:" + title + "\t" + genres + "\t" + str(sim_score))

# debug函数：debug推荐结果
def debug_recomresult(recom_result,item_info):
    """
    debug recomresult
    :param recom_result:key userid value:dict,value_key:itemid,value_value:recom_score
    :param item_info:dict,key itemid value:[title,genre]
    :return:
    """
    # 指定一个debug的userid：即可得到给 user 1 推荐的 Items
    user_id = "1"

    if user_id not in recom_result:
        print("invalid result")
        return
    for zuhe in sorted(recom_result[user_id].items(),key=operator.itemgetter(1),reverse=True):
        # 给zuhe定义数据类型
        itemid,score = zuhe
        if itemid not in item_info:
            continue
        print(",".join(item_info[itemid]) + "\t" + str(score))


def main_flow():
    """
    1,计算得到item的相似度
    2，根据相似度进行推荐
    3, 计算召回率和准确率
    """

    # 主功能区
    # step1:获得用户的点击序列    【获取到公式升级2的时间差】
    user_click, user_click_time = reader.get_user_click("../data/split/rat_train.txt")

    item_info = reader.get_item_info("../data/movies.txt")

    # step2:根据用户的点击序列，得到Item的相似度
    # sim_info = cal_item_sim(user_click)

    # 【升级2】在计算item的相似度的时候需要用到user_click_time
    sim_info = cal_item_sim(user_click,user_click_time)

    # debug itemsim使用
    debug_itemsim(item_info,sim_info)

    # step3:根据用户的点击序列以及item的sim_info,去计算推荐结果
    recom_result = cal_recom_result(sim_info,user_click)

    # print(recom_result["1"])

    # debug recomresult使用
    debug_recomresult(recom_result, item_info)

    # 计算评测指标(recom_result即是[user,itemid])
    # 1.获取test的用户点击序列
    test_userClick = reader.get_user_click("../data/split/rat_test.txt")[0]

    # 2.进入函数Recall\Precision\Coverage\Populartity
    recall = evaluation.Recall(recom_result, test_userClick)
    precision = evaluation.Precision(recom_result, test_userClick)
    coverage = evaluation.Coverage(recom_result, item_info)
    populartity = evaluation.Populartity(recom_result)

    print("recall（召回率）:" + str(recall) + "\t" + "precision（准确率）:" + str(precision))
    print("coverage（覆盖率）:" + str(coverage) + "\t" + "populartity（新颖度）:" + str(populartity))


if __name__=="__main__":
    main_flow()

