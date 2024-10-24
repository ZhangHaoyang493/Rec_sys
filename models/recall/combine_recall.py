from tqdm import tqdm
import pickle

def combine_recall_results(multi_recall_results, weight_dict=None, topk=25):
    final_recall_items_dict = {}
    
    # 将所有的分数归一化到0-1之间
    def norm_user_recall_items_sim(sorted_item_list):
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_score = sorted_item_list[-1][1]
        max_score = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_score > 0:
                # 如果所有的物品分数相同，这个时候max=min，所有的物品都给1分（满分）
                norm_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 1.0
            # 如果所有的物品都是热门物品，这个时候所有的score都是负数
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
    
        
    for method, recall_res in tqdm(multi_recall_results.items()):
        weight = 1 if weight_dict is None else  weight_dict[method]
        for user_id, recall_list in recall_res.items():
            recall_list = norm_user_recall_items_sim(recall_list)
            
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in recall_list.items():
                
                
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += score * weight
                
    for user, recall_res in final_recall_items_dict.items():
        final_recall_items_dict[user] = sorted(list(recall_res.items()), key=lambda x: x[1], reverse=True)[:topk]
    
    pickle.dump(final_recall_items_dict, open('/data/zhy/recommendation_system/Rec_sys/data/final_recall_res.pkl', 'wb'))