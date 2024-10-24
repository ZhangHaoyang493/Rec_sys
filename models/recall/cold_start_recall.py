# 冷启动必须召回没有被任何用户点击过的物品
# 先基于embedding召回一些物品，从中选出那些从没有被用户点击的物品
# 然后在这些物品中按照特定规则筛选出合适的物品

from embedding_recall import embedding_search_item
from CF_simple import item_cf_recall
from utils.data_utils import *
import pickle
from tqdm import tqdm
import pandas as pd
import datetime


def get_embedding_recall(all_click_df):
    user_embedding_recall_dict = {}
    user_item_time_dict = get_user_item_dict(all_click_df)
    with open('cache/item_sim_dict_by_embedding.pkl', 'rb') as f:
        i2i_dict = pickle.load(f)

    topk_number = 150
    recall_item_number = 100
    item_topk = get_top_k_items(all_click_df, topk_number)
    
    for user in tqdm(all_click_df['user'].unique()):
        user_embedding_recall_dict[user] = item_cf_recall(user, user_item_time_dict, i2i_dict, item_topk, recall_item_number)

    pickle.dump(user_embedding_recall_dict, open('cache/cold_start_init_recall_items.pkl', 'wb'))    


def cold_start_recall(embedding_recall_dict, user_hist_item_types_dict, user_hist_item_words_dict,
                      user_last_read_item_created_time_dict, item_type_dict, item_words_dict,
                      item_created_time_dict, click_article_ids_set, recall_item_num):
    cold_start_recall_dict = {}
    for user, items in tqdm(embedding_recall_dict.items()):
        cold_start_recall_dict.setdefault(user, [])
        for item, score in items.items():
            # 获取用户历史的阅读的文章type的set
            user_hist_item_type_set = user_hist_item_types_dict[user]
            # 获取历史阅读文章平均字数信息
            user_hist_mean_words = user_hist_item_words_dict[user]
            # 获取用户最后一次读的文章的创建时间
            user_last_read_item_created_time = user_last_read_item_created_time_dict[user]
            user_last_read_item_created_time = datetime.fromtimestamp(user_last_read_item_created_time)

            # 获取当前召回的文章信息
            item_type = item_type_dict[item]
            item_words = item_words_dict[item]
            item_created_time = item_created_time_dict[item]
            item_created_time = datetime.fromtimestamp(item_created_time)

            # 如果item的类型不在用户以前读过的新闻文章类型中
            # 或者这个item已经被点击过（这种情况下已经不是冷启动了）
            # 或者文章的单词数量和用户历史度过的单词的平均数量相差超过200
            # 或者和用户最后读的一篇新闻的相差时间已经超过了90天
            if item_type not in user_hist_item_type_set or \
                item in click_article_ids_set or \
                abs(item_words - user_hist_mean_words) > 200 or \
                abs((item_created_time - user_last_read_item_created_time).days) > 90:
                continue
            
            cold_start_recall_dict[user].append((item, score))
    cold_start_recall_dict = {k: sorted(v, key=lambda x: x[1], reverse=True)[:recall_item_num] for k, v in cold_start_recall_dict.items()}

    pickle.dump(cold_start_recall_dict, open('/data/zhy/recommendation_system/Rec_sys/data/cold_start_recall.pkl', 'wb'))


##### Help function

def get_user_hist_item_types_dict(user_df: pd.DataFrame, article_df: pd.DataFrame):
    
    article_df.rename(columns={'article_id': 'click_article_id'}, inplace=True)
    res = user_df.merge(article_df, on='click_article_id').groupby('user_id')['category_id'].agg(set).reset_index()
    res = dict(zip(res['user_id'], res['category_id']))
    return res

def get_user_hist_item_words_dict(user_df: pd.DataFrame, article_df: pd.DataFrame):
    
    article_df.rename(columns={'article_id': 'click_article_id'}, inplace=True)
    res = user_df.merge(article_df, on='click_article_id').groupby('user_id')['words_count'].agg('mean').reset_index()
    res = dict(zip(res['user_id'], res['words_count']))
    return res

def get_user_last_read_item_created_time_dict(user_df: pd.DataFrame, article_df: pd.DataFrame):
    # user_df = user_df.sort_values(by=['user_id', 'click_timestamp'])
    article_df.rename(columns={'article_id': 'click_article_id'}, inplace=True)
    res = user_df.merge(article_df, on='click_article_id').sort_values(by=['user_id', 'click_timestamp']).groupby('user_id')[['created_at_ts']].apply(lambda x : x.iloc[-1]).reset_index()
    res = dict(zip(res['user_id'], res['created_at_ts']))
    return res

def get_item_type_dict(article_df: pd.DataFrame):
    return dict(zip(article_df['article_id'], article_df['category_id']))

def get_item_words_dict(article_df: pd.DataFrame):
    return dict(zip(article_df['article_id'], article_df['words_count']))

def get_item_created_time_dict(article_df: pd.DataFrame):
    return dict(zip(article_df['article_id'], article_df['created_at_ts']))

def get_click_article_ids_set(user_df: pd.DataFrame):
    return set(user_df['click_article_id'].unique())



if __name__ == '__main__':
    print(get_click_article_ids_set(pd.read_csv('/data/zhy/recommendation_system/Rec_sys/data/train_click_log.csv')))