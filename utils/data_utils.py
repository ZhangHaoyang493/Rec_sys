import pandas as pd
import numpy as np
import pickle
import os
import os.path as osp

cache_path = '/data/zhy/recommendation_system/Rec_sys/cache'

def get_cache(file_name):
    file_path = osp.join(cache_path, file_name)
    if osp.exists(file_path):
        with open(file_path, 'rb') as f:
            file = pickle.load(f)
            return file
    else:
        return None
    
def save_to_cache(file_name, data):
    file_path = osp.join(cache_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_user_data(is_all_data=False):
    train_df = pd.read_csv('data/train_click_log.csv')
    if is_all_data:
        test_df = pd.read_csv('data/testA_click_log.csv')
        return pd.concat([train_df, test_df])
    return train_df

def get_article_data():
    article_df = pd.read_csv('/data/zhy/recommendation_system/Rec_sys/data/articles.csv')
    article_df = article_df.rename(columns={'article_id': 'click_article_id'})

    return article_df

def get_article_embedding_data():
    """返回文章-embedding字典

    Returns:
        字典: {文章id: embedding}
    """
    item_embedding_dict = get_cache('article_embedding.pkl')
    if item_embedding_dict is not None:
        return item_embedding_dict
    
    article_emb_df = pd.read_csv('data/articles_emb.csv')
    emb_cols = [x for x in article_emb_df.columns if 'emb' in x]

    article_embedding = article_emb_df[emb_cols].to_numpy()#.to_list()
    article_embedding = article_embedding / np.linalg.norm(article_embedding, axis=1, keepdims=True)

    item_embedding_dict = dict(zip(article_emb_df['article_id'], article_embedding))

    save_to_cache('article_embedding.pkl', item_embedding_dict)
    
    return item_embedding_dict

# emb = get_article_embedding_data()
# emb[0].shape

def get_user_item_dict(user_df: pd.DataFrame) -> dict:
    """返回
    {user_id: [(article_id, time), (article_id, time), ...}}

    Args:
        user_df (_type_): _description_
    """
    def make_item_time_dict(x):
        return list(zip(x['click_article_id'], x['click_timestamp']))
    
    res = user_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_dict(x)).reset_index().rename(columns={0: 'item_user_list'})
    return dict(zip(res['user_id'], res['item_user_list']))

def get_item_user_dict(user_df: pd.DataFrame) -> dict:
    """{item_id: [(user_id, time), (user_id, time), ...]}

    Args:
        user_df (pd.DataFrame): _description_

    Returns:
        dict: _description_
    """
    def make_user_time_dict(x):
        return list(zip(x['user_id'], x['click_timestamp']))
    
    res = user_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: make_user_time_dict(x)).reset_index().rename(columns={0: 'user_time_list'})
    return dict(zip(res['click_article_id'], res['user_time_list']))

def get_last_and_history_click(user_df: pd.DataFrame):
    user_df = user_df.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = user_df.groupby('user_id').tail(1)
    
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]
        
    click_hist_df = user_df.groupby('user_id').apply(lambda x: hist_func(x)).reset_index(drop=True)
    return click_last_df, click_hist_df

def get_item_info_dict(article_df: pd.DataFrame):
    # 归一化时间节点
    max_min_scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
    article_df['created_at_ts'] = article_df['created_at_ts'].apply(max_min_scaler)
    
    item_type_dict = dict(zip(article_df['click_article_id'], article_df['category_id']))
    words_count_dict = dict(zip(article_df['click_article_id'], article_df['words_count']))
    created_time_dict = dict(zip(article_df['click_article_id'], article_df['created_at_ts']))

    return item_type_dict, words_count_dict, created_time_dict


def get_user_history_click_item_info(user_df: pd.DataFrame):
    if os.path.exists('cache/user_category_dict.pkl'):
        with open('cache/user_category_dict.pkl', 'rb') as f:
            user_category_dict = pickle.load(f)
    else:
        user_category_df = user_df.groupby('user_id')['category_id'].apply(lambda x: set(x)).reset_index()#.rename(columns={0: 'category_id_set'})
        user_category_dict = dict(zip(user_category_df['user_id'], user_category_df['category_id']))
        with open('cache/user_category_dict.pkl', 'wb') as f:
            pickle.dump(user_category_dict, f)
    
    if os.path.exists('cache/user_click_item_dict.pkl'):
        with open('cache/user_click_item_dict.pkl', 'rb') as f:
            user_click_item_dict = pickle.load(f)
    else:
        user_click_item_df = user_df.groupby('user_id')['click_article_id'].apply(lambda x: set(x)).reset_index()#.rename(columns={0: 'category_id_set'})
        user_click_item_dict = dict(zip(user_click_item_df['user_id'], user_click_item_df['click_article_id']))
        with open('cache/user_click_item_dict.pkl', 'wb') as f:
            pickle.dump(user_click_item_dict, f)


    if os.path.exists('cache/user_words_dict.pkl'):
        with open('cache/user_words_dict.pkl', 'rb') as f:
            user_words_dict = pickle.load(f)
    else:
        user_words_df = user_df.groupby('user_id')['words_count'].apply(lambda x: np.mean(x)).reset_index()#.rename(columns={0: 'category_id_set'})
        user_words_dict = dict(zip(user_words_df['user_id'], user_words_df['words_count']))
        with open('cache/user_words_dict.pkl', 'wb') as f:
            pickle.dump(user_words_dict, f)

    if os.path.exists('cache/user_last_click_dict.pkl'):
        with open('cache/user_last_click_dict.pkl', 'rb') as f:
            user_last_click_dict = pickle.load(f)
    else:
        user_df_sort = user_df.sort_values(by=['user_id', 'click_timestamp'])
        user_last_click_df = user_df_sort.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()

        max_min_scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
        # print(np.min(user_last_click_df['created_at_ts']))
        user_last_click_df['created_at_ts'] = user_last_click_df[['created_at_ts']].apply(max_min_scaler)
        user_last_click_dict = dict(zip(user_last_click_df['user_id'], user_last_click_df['created_at_ts']))
        with open('cache/user_last_click_dict.pkl', 'wb') as f:
                pickle.dump(user_last_click_dict, f)



    return user_category_dict, user_click_item_dict, user_words_dict, user_last_click_dict


def get_top_k_items(user_df: pd.DataFrame, topk=50):
    # topk_dict = {}
    return user_df['click_article_id'].value_counts()[:topk]

if __name__ == '__main__':
    user_data = get_user_data()
    item_data = get_article_data()
    a, b, c, d = get_user_history_click_item_info(user_data.merge(item_data, on='click_article_id'))
    d
    # user_data.merge(item_data, on='click_article_id')