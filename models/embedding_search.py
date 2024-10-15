import os, sys
sys.path.append('/data/zhy/recommendation_system/Rec_sys')

from utils.data_utils import *
import pandas as pd
import numpy as np
import math
import pickle
import faiss
from tqdm import tqdm
from utils.utils import *

def embedding_search_item(item_embedding_df: pd.DataFrame, topk=5):
    item_sim_dict = get_cache('item_sim_dict_by_embedding.pkl')
    if item_sim_dict is not None:
        return item_sim_dict
    
    index_to_id_dict = dict(zip(item_embedding_df.index, item_embedding_df['article_id']))
    
    embedding_col = [x for x in item_embedding_df.columns if 'emb' in x]
    embeddings = np.ascontiguousarray(item_embedding_df[embedding_col].values, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 建立faiss索引并搜索
    data_base = faiss.IndexFlatIP(embeddings.shape[1])
    data_base.add(embeddings)
    sim, idx = data_base.search(embeddings, topk)
    
    # 每一个键值对如下：
    # key: 一个物品
    # value: 一个字典，保存了和key最相似的topk-1个物品的  物品id:相似度 的键值对
    item_sim_dict = {}
    for index, similar, ix in tqdm(zip(range(len(embeddings)), sim, idx)):
        item_id = index_to_id_dict[index]
        item_sim_dict.setdefault(item_id, {})
        # 去掉第一个元素，因为第一个元素就是自身
        for s, i in zip(similar[1:], ix[1:]):
            sim_item_id = index_to_id_dict[i]
            item_sim_dict[item_id].setdefault(sim_item_id, 0)
            item_sim_dict[item_id][sim_item_id] += s
            
    save_to_cache('item_sim_dict_by_embedding.pkl', item_sim_dict)
    return item_sim_dict

if __name__ == '__main__':
    res = embedding_search_item(pd.read_csv('data/articles_emb.csv'))
    display_dict(res)
    
    
    