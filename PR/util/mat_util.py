# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 15:15
# @Author  : Cathy
# @FileName: mat_util.py
# @Software: PyCharm
# mat util for personal rank algo

"""
Step1:根据之前得到的user_item二分图，得到 M 矩阵
Step2:根据 M 矩阵得到单位阵r0*(1-alpha)
Step3:对第二步得到的矩阵求逆
"""
from  __future__ import division

import numpy as np
from scipy.sparse import coo_matrix

# 引入得到用户二分图的函数
import sys
sys.path.append("../util")
import util.read as read

def graph_to_m(graph):
    """
    :param graph: user item graph
    :return: a coo_matrix 一个稀疏矩阵 sparse mat M
                a list,total user item point
                a dict,map all the point to row index
    """

    # 定义所有的顶点,并用字典来存储;即可知道在构建 M 矩阵的时候，每一行对应的是哪个顶点
    vertex = list(graph.keys())
    address_dict = {}

    # 指定行和列
    total_len = len(vertex)

    for index in range(len(vertex)):
        address_dict[vertex[index]] = index

    # 对于coo这种稀疏矩阵，需要定义三个数组，分别存储对应的行索引、列索引和对应的数值
    row = []
    col = []
    data = []

    for element_i in graph:
        weight = round(1/len(graph[element_i]),3)
        row_index = address_dict[element_i]
        for element_j in graph[element_i]:
            col_index = address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)

    # 初始化稀疏矩阵，同时讲三个数组，转换成npArray的数据结构
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data,(row,col)),shape=(total_len,total_len))

    # 需要返回 M 矩阵、所有的顶点，以及所有的顶点在图中的位置
    return m,vertex,address_dict


def mat_all_point(m_mat,vertex,alpha):
    """
    get E-alpha*m_mat.T
    :param m_mat:
    :param vertex: total item and user point 所有的顶点
    :param alpha: the prob for random walking 选的随机游走的概率
    :return: a sparse
    """
    # 利用 coo 的方式去初始化单位阵
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data,(row,col)),shape=(total_len,total_len))
    print(eye_t.todense())

    # 输出稀疏矩阵中的结果
    return (eye_t.tocsr() - alpha * m_mat.tocsr().transpose())

if __name__ == "__main__":
    graph = read.get_graph_from_data("../data/log.txt")
    m,vertex,address_dict = graph_to_m(graph)

    # 测试第一个函数
    # print(address_dict)
    # print(m.tocsr())

    # 测试第二个函数
    print(mat_all_point(m,vertex,0.8))