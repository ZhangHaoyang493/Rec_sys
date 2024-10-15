import os, sys
sys.path.append('/data/zhy/recommendation_system/Rec_sys')

from utils.data_utils import *
import pandas as pd
import numpy as np
import math
import pickle

def cal_i2i_sim_matrix(user_item_time_dict):
    item_similar = get_cache('simple_item_similar_matrix.pkl')
    if item_similar is not None:
        return item_similar
    
    item_similar = {}
    item_cnt = {}
    for user, user_hist in user_item_time_dict.items():
        for item, t in user_hist:
            item_similar.setdefault(item, {})
            item_cnt.setdefault(item, 0)
            item_cnt[item] += 1
            for item_, t_ in user_hist:
                # 下面这一行代码一定要放到最前面，否则自身和自身的相似度就没有定义
                item_similar[item].setdefault(item_, 0)
                if item == item_:
                    continue
                item_similar[item][item_] += 1
            
    for item, simi in item_similar.items():
        for item_, simi_ in simi.items():
            item_similar[item][item_] /= math.sqrt(item_cnt[item] * item_cnt[item_])
    
    save_to_cache('simple_item_similar_matrix.pkl', item_similar)
    
    
def cal_u2u_sim_matrix(item_user_item_dict):
    user_similar = get_cache('simple_user_similar_matrix.pkl')
    if user_similar is not None:
        return user_similar
    
    user_similar = {}
    user_cnt = {}
    
    for item, item_hist in item_user_item_dict.items():
        for user, t in item_hist.items():
            user_similar.setdefault(user, {})
            user_cnt.setdefault(user, 0)
            user_cnt[user] += 1
            for user_, t_ in item_hist.items():
                # 下面这一行代码一定要放到最前面，否则自身和自身的相似度就没有定义
                user_similar[user].setdefault(user_, 0)
                if user == user_:
                    continue
                user_similar[user][user_] += 1
    
    for i in user_similar.keys():
        for j in user_similar[i].keys():
            user_similar[i][j] /= math.sqrt(user_cnt[i] * user_cnt[j])
    
    save_to_cache('simple_user_similar_matrix.pkl', user_similar)
            
if __name__ == '__main__':
    user_df = get_user_data()
    user_item_time = get_user_item_dict(user_df)
    matrix = cal_i2i_sim_matrix(user_item_time)
    print(matrix)