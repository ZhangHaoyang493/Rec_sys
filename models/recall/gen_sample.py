import pandas as pd
from tqdm import tqdm
import numpy as np
import random

# 正样本：用户的历史样本
# 负样本：不在用户历史样本中的文章随机选择一些
# 预测什么，预测用户历史样本中最后一次点击的样本
def gen_train_test_samples(user_hist_data_df: pd.DataFrame, neg_sample_num: int = 1):
    """_summary_

    Args:
        user_hist_data_df (pd.DataFrame): 
        neg_sample_num (int, optional): 一个正样本对应几个负样本. Defaults to 1.
    """
    user_hist_data_df.sort_values(by=['click_timestamp'], inplace=True)
    # 将用户历史中所有出现的物品提取出来
    # 那么后续选择负样本时，就是从所有用户看到过的所有物品但是当前用户没看到的物品中随机抽取
    item_ids = user_hist_data_df['click_article_id'].unique()
    
    train_set, test_set = [], []
    for user_id, user_hist in tqdm(user_hist_data_df.groupby('user_id')):
        hist_list = user_hist['click_article_id'].tolist()
        
        if neg_sample_num > 0:
            candidate_negative_samples = list(set(item_ids) - set(hist_list))
            neg_list = np.random.choice(candidate_negative_samples, 
                                        size=len(hist_list) * neg_sample_num,
                                        replace=False)
        
        # 如果用户历史只有最后一个，那么手动添加到训练集中，
        # 否则这个用户不会在训练集中出现
        if len(hist_list) == 1:
            train_set.append((user_id, [hist_list[0]], hist_list[0], 1, len(hist_list)))
            test_set.append((user_id, [hist_list[0]], hist_list[0], 1, len(hist_list)))
            
        # 滑窗法构建正负样本
        for i in range(1, len(hist_list)):
            hist_now = hist_list[:i]
            
            if i != len(hist_list) - 1:
                # 这里将hist_now进行了反转，可能是想让最近刚浏览的物品影响大一些
                # 正样本 (用户id，用户观看历史，预测目标（也就是当前最后浏览的物品），标签，观看历史长度)
                train_set.append((user_id, hist_now[::-1], hist_list[i], 1, len(hist_now)))
                for j in range(neg_sample_num):
                    # 负样本
                    train_set.append((user_id, hist_now[::-1], neg_list[i * neg_sample_num + j], 0, len(hist_now)))
            
            # 最后一个样本用作测试用
            else:
                test_set.append((user_id, hist_now[::-1], hist_list[i], 0, len(hist_now)))
                
        # 打乱
        random.shuffle(train_set)
        random.shuffle(test_set)
        
        return train_set, test_set
    