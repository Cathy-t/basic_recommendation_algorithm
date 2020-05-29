# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 15:49
# @Author  : Cathy
# @FileName: usercf.py
# @Software: PyCharm
from __future__ import division
import math
import operator
import sys
sys.path.append("../util")
import util.reader as reader

import util.evaluation_metrics_test as evaluation

# 数据结构转化函数：将用户的点击转化为item被user点击
def transfer_user_click(user_click):
    """
    :param user_click:dict,key userid,value:[itemid1,itemid2]
    :return:dict,key itemid,value:[userid1,userid2]
    """
    item_click_by_user = {}
    for user in user_click:
        item_list = user_click[user]
        for itemid in item_list:
            item_click_by_user.setdefault(itemid, [])
            item_click_by_user[itemid].append(user)
    return item_click_by_user

# 基本贡献值
def base_contribution_score():
    """
    base usercf user contribution score
    """
    return 1

# 【升级版1】中的贡献值
def update_contribution_score(item_user_click_count):
    """
    usercf user contribution score v1
    :param item_user_click_count: item被多少个用户点击过 how many users have clicked this item
    :return: 贡献度的得分 contribution score
    """
    return 1/math.log10(1 + item_user_click_count)

# 【升级版2】中的贡献值
def update_two_contribution_score(click_time_one, click_time_two):
    """
    usercf user contribution score v2
    :param click_time_one , click_time_two: 不同user对同一item的点击时间
    :return:contribution score
    """
    # 计算时间差
    delta_time = abs(click_time_two - click_time_one)
    # 需将秒级别的转化成天级别的，使影响变大
    # 即一天有多少秒
    norm_num = 60 * 60 * 24
    delta_time = delta_time/norm_num

    return 1/(1 + delta_time)

# 计算用户相似度
# user_click_time即在【公式升级2】中使用
def cal_user_sim(item_click_by_user,user_click_time):
    """
    get user sim info
    :param item_click_by_user:dict,key itemid,value:[userid1,userid2]
    :return:dict,key itemid,value:dict ,value_key:itemid_j,value_value:simscore
    """
    co_appear = {}
    user_click_count = {} # 表示用户点击的次数
    # 出现在同一个Item点击的User序列中的两个user,他们的贡献都会+1
    for itemid, user_list in item_click_by_user.items():
        for index_i in range(0, len(user_list)):
            user_i = user_list[index_i]
            user_click_count.setdefault(user_i, 0)
            user_click_count[user_i] += 1

            # 【升级版2】中 user_i 对item 的点击时间
            if user_i + "_" + itemid not in user_click_time:
                click_time_one = 0
            else:
                click_time_one = user_click_time[user_i + "_" + itemid]

            for index_j in range(index_i + 1, len(user_list)):
                user_j = user_list[index_j]

                # 【升级版2】中 user_j 对item 的点击时间
                if user_j + "_" + itemid not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user_j + "_" + itemid]

                co_appear.setdefault(user_i, {})
                co_appear[user_i].setdefault(user_j, 0)

                # co_appear[user_i][user_j] += base_contribution_score()
                # 【升级版1】降低那些异常活跃物品对用户相似度的贡献
                # co_appear[user_i][user_j] += update_contribution_score(len(user_list))
                # 【升级版2】不同用户对同一item行为的时间段不同应给予时间惩罚
                co_appear[user_i][user_j] += update_two_contribution_score(click_time_one,click_time_two)

                co_appear.setdefault(user_j, {})
                co_appear[user_j].setdefault(user_i, 0)
                # co_appear[user_j][user_i] += base_contribution_score()
                # co_appear[user_j][user_i] += update_contribution_score(len(user_list))
                co_appear[user_j][user_i] += update_two_contribution_score(click_time_one, click_time_two)

    user_sim_info = {}
    # 对计算出的相似度，先排序，再输出
    user_click_count_sorted = {}
    for user_i, relate_user in co_appear.items():
        user_sim_info.setdefault(user_i, {})
        for user_j, cotime in relate_user.items():
            # 将user_i对user_j的相似度初始化为0
            user_sim_info[user_i].setdefault(user_j, 0)
            # 实际数值
            user_sim_info[user_i][user_j] = cotime/math.sqrt(user_click_count[user_i] * user_click_count[user_j])
    # 对计算出的相似度，先排序，再输出
    for user in user_sim_info:
        user_click_count_sorted[user] = sorted(user_sim_info[user].items(),key=operator.itemgetter(1),reverse=True)

    return user_click_count_sorted

# 得到推荐结果的函数
def cal_recom_result(user_click,user_sim):
    """
    recom by usercf algo

    :param user_click:dict,用户的点击序列 key userid, value [itemid1,itemid2]
    :param user_sim:经过排序后的用户的相似度矩阵 key:userid value:list,[(useridj,score1),(useridk,score2)]
    :return:dict,key userid value:dict,value_key:itemid1,value_value:recom_score
    """
    recom_result = {}
    topk_user = 3
    item_num = 5
    # 需要过滤掉该用户已点击过的物品
    for user, item_list in user_click.items():
        # 定义一个临时的字典，来完成过滤
        tmp_dict = {}
        for itemid in item_list:
            tmp_dict.setdefault(itemid, 1)

        recom_result.setdefault(user, {})

        for zuhe in user_sim[user][:topk_user]:
            userid_j, sim_score = zuhe
            # 如果user_j 不在点击序列里面，则将其过滤掉；若在，则完成推荐
            if userid_j not in user_click:
                continue
            for itemid_j in user_click[userid_j][:item_num]:
                recom_result[user].setdefault(itemid_j,sim_score)
    return recom_result

def debug_user_sim(user_sim):
    """
    print user sim result
    :param user_sim: key userid value:[(userid1,score1),(userid2,score2)]
    """
    topk = 80
    #选择一个固定的用户进行打印
    fix_user = "1"
    if fix_user not in user_sim:
        print("invalid user")
        return
    for zuhe in user_sim[fix_user][:topk]:
        userid, score = zuhe

        print(fix_user + "\tsim_user:" + userid + "\t" + str(score))

def debug_recom_result(item_info, recom_result):
    """
    print recom result for user
    :param item_info: key itemid value:[title,genres]
    :param recom_result: key userid ,value:dict,value_key:itemid,value_value:recom_score
    """

    fix_user = "1"
    if fix_user not in recom_result:
        print("invalid user for recoming result")
        return

    for itemid in recom_result["1"]:
        if itemid not in item_info:
            continue
        recom_score = recom_result["1"][itemid]

        print("recom_result:" + ",".join(item_info[itemid]) + "\t" + str(recom_score))


def main_flow():
    """
    main_flow
    1,计算用户的相似度矩阵
    2，根据user相似度矩阵给user推荐物品
    """

    # step1:获得用户的点击序列
    user_click, user_click_time = reader.get_user_click("../data/split/rat_train.txt")

    item_info = reader.get_item_info("../data/movies.txt")

    # step2:同一个用户点击的item有哪些，根据此贡献值进行计算，所以需要将user_click转化成item_click_by_user，并计算其相似度
    item_click_by_user = transfer_user_click(user_click)

    # user_sim = cal_user_sim(item_click_by_user)
    # 【公式升级2】中即需要考虑到用户对同一物品点击的时间差，时间差越小，贡献度越高
    user_sim = cal_user_sim(item_click_by_user, user_click_time)

    # debug相似度的函数
    debug_user_sim(user_sim)

    # step3:根据用户的相似度矩阵，及点击序列，即可得到对用户的推荐结果
    recom_result = cal_recom_result(user_click, user_sim)

    # debug推荐结果的函数
    debug_recom_result(item_info, recom_result)

    print(recom_result["1"])

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


if __name__ == "__main__":
    main_flow()