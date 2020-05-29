# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 16:19
# @Author  : Cathy
# @FileName: reader.py
# @Software: PyCharm

import os

# 对文件中公共信息的提取

def get_user_click(rating_file):
     """
     get user click list
     :param rating_file:input file
     :return: dict;key:userid,value"[itemid1,itemid2]
     """

     # 判断文件是否存在
     if not os.path.exists(rating_file):
         return {},{}

     # 打开一个文件
     fp = open(rating_file)

     # 声明一个计数器，便于读取文件中的有效数据
     num = 0

     # 声明一个返回所需的数据结构
     user_click = {}

     # 【公式升级2所需数据结构：需要获取到用户的点击时间差】
     user_click_time = {}

     # 按行处理文件
     for line in fp:
         if num == 0:
             num += 1
             continue
         item = line.strip().split(',')
         if len(item) < 4:
            continue
         [userid,itemid,rating,timestamp] = item

         # 【升级2所用】
         if userid + '_' +itemid not in user_click_time:
             user_click_time[userid + '_' +itemid] = int(timestamp)

         # 若评分大于三分记为喜欢该电影（阈值）
         if float(rating) < 3.0:
             continue

         if userid not in user_click:
             user_click[userid] = []
         user_click[userid].append(itemid)

     # 处理完后关闭文件
     fp.close()

    # user_click_time为公式升级2时，所需的时间差
     return user_click,user_click_time

def get_item_info(item_file):
    """
    get item info[title,genres(类别)]
    :param item_file: input iteminfo file
    :return:a dict ,key itemid,value[title,genres]
    """

    if not os.path.exists(item_file):
        return {}
    num = 0

    item_info = {}

    fp = open(item_file,errors='ignore')
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        # 对数据进行处理，即若分割后的元素>3个
        if len(item) == 3:
            [itemid,title,genres] = item
        elif len(item) > 3:
            itemid = item[0]
            # item[-1]即表示最后一列
            genres = item[-1]
            # ",".join(item[1:-1]) 表示第一列到最后一列的拼接
            title = ",".join(item[1:-1])

        if itemid not in item_info:
            item_info[itemid] = [title,genres]
    fp.close()
    return item_info


# 测试功能
if __name__ == "__main__":
    # 检测第一个函数 get_user_click
    # user_click = get_user_click("../data/ratings.txt")
    # print(len(user_click))
    # print(user_click["1"])

    # 检测第二个函数 get_item_info
    item_info = get_item_info("../data/movies.txt")
    print(len(item_info))
    print(item_info["11"])



