# -*- coding: utf-8 -*-
# @Time    : 2019/3/18 17:40
# @Author  : Cathy
# @FileName: personal_rank.py
# @Software: PyCharm

from __future__ import division

import sys
sys.path.append("../util")
import util.read as read

import operator
import util.mat_util as mat_util

# 解稀疏矩阵的方程所需使用的模块 gmres
from scipy.sparse.linalg import gmres

import numpy as np

def personal_rank(graph,root,alpha,iter_num,recom_num=10):
    """
    :param graph: user item graph 之前得到的user和item的图结构
    :param root: the fixed user for which to recom 将要给哪个user推荐
    :param alpha: the prob to go to random walk 以alpha的概率选择向下游走，以1-alpha的概率选择回到起点
    :param iter_num:  iteration num 迭代次序
    :param recom_num: recom item num 推荐的结果
    :return: a dict: key itemid,value pr值 字典的长度即为指定的推荐的item的个数
    """

    # 定义一个数据结构来存储所有的顶点对于root顶点的pr值
    rank = {}
    # pr算法中，pr值的初始条件中：除root顶点外其余顶点的pr值均为0
    rank = {point:1 for point in graph}
    rank[root] = 1

    # 定义一个输出数据结构
    recom_result = {}

    for iter_index in range(iter_num):

        # 初始化一个临时的数据结构，此数据结构用于存储该迭代轮次下，其余顶点对root顶点的pr值
        tmp_rank = {}
        tmp_rank = {point:0 for point in graph}

        # PR算法公式书写：分为上下两部分
        # 在上部分中，如果该顶点不是root顶点，它的pr值就是所有连接到该顶点的顶点
        # 将自己的pr值，以1/n的概率贡献到该顶点上（n就是连接到该顶点的顶点的出度）
        for out_point,out_dict in graph.items():
            for inner_point,value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha * rank[out_point]/len(out_dict),4)

                if inner_point == root:
                    tmp_rank[inner_point] += round(1-alpha,4)

        # 如果该迭代轮次下的临时的数据结构和装载所有顶点对root顶点pr值的数据结构完全相同时，即为迭代充分
        # 此时可提前结束迭代
        if tmp_rank == rank:

            # 是否是迭代完成了iter_num次，还是迭代到中间的部分就完成了收敛
            print("out" + str(iter_index))

            break
        # 若不相同，则需将本轮次最新迭代出的root顶点的pr值，赋值给rank
        rank = tmp_rank

    # rank迭代完成后，对rank中的pr值进行排序，并过滤掉其中的user顶点和root顶点已经行为过的item，这样就能得到最终的推荐结果

    # 定义一个计数器,帮助记录如果推荐的item的数目达到了要求，就可以返回
    right_num = 0

    # Step1:排序
    for zuhe in sorted(rank.items(),key=operator.itemgetter(1),reverse=True):
        point,pr_score = zuhe[0],zuhe[1]

        # 如果该顶点不是item顶点，则需要过滤掉
        if len(point.split('_')) < 2:
            continue

        # 如果该顶点是item顶点，且被root顶点行为过，仍需要过滤掉
        if point in graph[root]:
            continue

        recom_result[point] = pr_score
        right_num += 1

        if right_num > recom_num:
            break

    return recom_result


def personal_rank_mat(graph,root,alpha,recom_num=10):
    """
    :param graph: user item graph 用户物品的二分图
    :param root:  the fix user to recom 固定用户推荐
    :param alpha: the prob to random walk 随机游走的概率
    :param recom_num: recom item num
    :return: a dict, key :itemid ,value:pr score

    线代相关知识：求矩阵的逆矩阵，即解线性方程 Ax = E （A*r = r0）
    """
    m, vertex, address_dict = mat_util.graph_to_m(graph)
    if root not in address_dict:
        return {}

    score_dict = {}
    recom_dict = {}

    # 求其逆，便可以得到推荐结果
    mat_all = mat_util.mat_all_point(m,vertex,alpha)

    # 首先得到root顶点的index,得到index的目的是为了获得r0矩阵
    index = address_dict[root]

    # 初始化r0矩阵
    initial_list = [[0] for row in range(len(vertex))]
    initial_list[index] = [1]
    r_zero = np.array(initial_list)
    # r_zero = np.concatenate(r_zero,axis=0)

    # 解线性方程,得到的是一个元组，其中tol指的是误差
    res = gmres(mat_all,r_zero,tol=1e-8)[0]

    for index in range(len(res)):
        # 首先判断该顶点是否是item顶点
        point = vertex[index]
        if len(point.strip().split("_")) < 2:
            continue
        # 若已经行为过，则也没有必要记录
        if point in graph[root]:
            continue

        score_dict[point] = round(res[index],3)

    # 讲pr值排序，返回推荐结果
    for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:recom_num]:
        point,score = zuhe[0],zuhe[1]
        recom_dict[point] = score
    return recom_dict


# personal rank 基础版本
def get_one_user_recom():
    """
    give one fix_user recom result
    """
    user = "1"
    alpha = 0.8
    graph = read.get_graph_from_data("../data/ratings.txt")
    iter_num = 100

    recom_result = personal_rank(graph,user,alpha,iter_num,100)

    return recom_result

    """
    item_info = read.get_item_info("../data/movies.txt")

    # 打印出用户感兴趣的item ，以便于分析结果
    for itemid in graph[user]:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])

    print("result---")
    for itemid in recom_result:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])
    """

# personal rank采用矩阵版本
def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "1"
    alpha = 0.8
    graph = read.get_graph_from_data("../data/ratings.txt")
    recom_result = personal_rank_mat(graph,user,alpha,100)
    return recom_result


if __name__ == "__main__":

    # 将两种方式进行对比
    recom_result_base = get_one_user_recom()
    recom_result_mat = get_one_user_by_mat()

    # 二种方式下的推荐结果有多少是相同
    num = 0
    for ele in recom_result_base:
        if ele in recom_result_mat:
            num += 1

    # 输出的num说明两种方式推荐出来的结果的相似度，99说明在top-N=100中，重合率很高，即两种方式效果一样
    print(num)