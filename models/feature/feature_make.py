import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils.data_utils import *

def train_val_split(all_click_df: pd.DataFrame, sample_user_nums: int):
    all_user_ids = all_click_df['user_id'].unique()
    
    val_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)
    
    click_val = all_click_df[all_click_df['user_id'].isin(val_user_ids)]
    click_train = all_click_df[~all_click_df['user_id'].isin(val_user_ids)]
    
    # 验证集中每个用户的最后一次点击为验证集的答案
    click_val = click_val.sort_values(by=['user_id', 'click_timestamp'])
    click_val_ans = click_val.groupby('user_id').tail(1)
    
    # 除了最后一次的点击数据作为答案，前面的作为输入
    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)
    # 有些用户只有一条点击记录，这样的用户只在ans中出现，不在click_val中出现，应该丢弃
    click_val_ans = click_val_ans[click_val_ans['user_id'].isin(click_val['user_id'].unique())]
    click_val = click_val[click_val['user_id'].isin(click_val_ans['user_id'].unique())]
    
    return click_train, click_val, click_val_ans



def get_hist_and_last_click(all_click: pd.DataFrame):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)
    
    def hist_func(x):
        if len(x) == 1:
            return x
        else:
            return x[:-1]
    
    click_hist_df = all_click.groupby('user_id').apply(lambda x : hist_func(x)).reset_index(drop=True)
    return click_hist_df, click_last_df

# 创建训练集，验证集，测试集
def get_train_val_test_raw_data(data_path, offline=True):
    test_data = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))
    if offline:
        all_data = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        click_train, click_val, val_ans = train_val_split(all_data, sample_user_nums=200)
    # 线上预测的话所有的训练集都可以拿过来训练，不需要划分测试集
    else:
        all_data = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        click_train = all_data
        click_val = None
        val_ans = None
    return click_train, click_val, test_data, val_ans


# 将召回列表转换为df
def recall_list_to_df(recall_list_dict):
    df_row_list = []
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])
    
    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)
    
    return recall_list_df


# 这里的label_df是用户最后一次阅读的文章组成的df，这些标签应该为1
def set_recall_label(recall_df: pd.DataFrame, label_df: pd.DataFrame, is_test=False):
    # 如果是测试集，那么就不需要给标签，为了保证一致性，标签给-1
    if is_test:
        recall_df['label'] = -1
        return recall_df
    
    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_df_ = recall_df.merge(label_df['user_id', 'sim_item', 'click_timestamp'], 
                                 on=['user_id', 'sim_item'], how='left')
    # 如果x是nan，说明在上述左连接的过程中，recall_df对应的user_id和sim_item在label_df没有对应的项
    # 这就说明用户并没有点击过这个文章，那就说明这个标签应该是0，反之是1
    recall_df_['label'] = recall_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_df_['click_timestamp']
    
    return recall_df_

# 进行负采样
# 正样本是召回的并且点击的，负样本是召回没有点击的
# sample_rate: 一个正样本对应几个负样本
# max_neg_sample: 一个正样本最多采样几个负样本
def neg_sample_recall_data(recall_df: pd.DataFrame, sample_rate=0.001, max_neg_sample=5):
    pos_data = recall_df[recall_df['label'] == 1]
    neg_data = recall_df[recall_df['label'] == 0]
    
    def neg_sample(x: pd.DataFrame):
        neg_num = len(x)
        # 最少采样一个
        sample_num = max(sample_rate * neg_num, 1)
        # 最多采样max_neg_sample
        sample_num = min(sample_num, max_neg_sample)
        return x.sample(sample_num, replace=True)
    
    
    # 对用户进行负采样，保证所有用户都有负样本
    neg_data_user = neg_data.groupby('user_id', group_keys=False).apply(lambda x: neg_sample(x))
    # 对物品进行负采样，保证所有的物品都有负样本
    neg_data_item = neg_data.groupby('sim_item', group_keys=False).apply(lambda x: neg_sample(x))
    
    # 可能两个负样本采样的data有重复
    all_neg_data = pd.concat([neg_data_user, neg_data_item])
    all_neg_data = all_neg_data.sort_values(['user_id', 'score']).drop_duplicates(subset=['user_id', 'sim_item'], keep='last')
    
    # 合并正负样本
    all_data = pd.concat([pos_data, all_neg_data])
    return all_data
    
    
# 获取训练数据，验证数据以及测试数据
# 返回数据格式
# user_id   sim_item  score label
def get_train_val_test_data(data_path, recall_df):
    # click_hist_df, click_last_df = get_hist_and_last_click(all_click)
    click_train, click_val, test_data, val_ans = get_train_val_test_raw_data(data_path)
    click_train_hist, click_train_last = get_hist_and_last_click(click_train)

    click_val_hist, click_val_last = click_val, val_ans
    
    click_test_hist, click_test_last = get_hist_and_last_click(test_data)
    
    # 获取训练数据中用户的召回列表
    train_recall_df = recall_df[recall_df['user_id'].isin(click_train_hist['user_id'].unique())]
    # 训练数据加上标签
    train_label_recall_df = set_recall_label(train_recall_df, click_train_last, is_test=False)
    train_label_recall_df = neg_sample_recall_data(train_label_recall_df)

    if click_val is not None:
        val_recall_df = recall_df[recall_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_label_recall_df = set_recall_label(val_recall_df, click_val_last, is_test=False)
        val_label_recall_df = neg_sample_recall_data(val_label_recall_df)
    else:
        val_label_recall_df = None

    test_recall_df = recall_df[recall_df['user_id'].isin(click_test_hist['user_id'].unique())]
    test_label_recall_df = set_recall_label(test_recall_df, click_test_last, is_test=False)

    return train_label_recall_df, val_label_recall_df, test_label_recall_df


# 将上面函数返回的带标签的recall_df转为字典
# {user_id : [(sim_item, score, label), ...], ...}
def recall_df_to_dict(recall_df: pd.DataFrame):
    def to_tuple_list(x):
        data = []
        for index, row in x.iterrows():
            data.append((row['sim_item'], row['score'], row['label']))
        return data
    
    recall_list = recall_df.groupby('user_id').apply(lambda x : to_tuple_list(x)).reset_index()
    return dict(list(recall_list['user_id'], recall_list[0]))


# 物品和用户的历史浏览之间的特征
# 对于该用户的每个召回商品， 
# 计算与上面最后N次点击商品的相似度的和(最大， 最小，均值)， 
# 时间差特征，相似性特征，字数差特征，与该用户的相似性特征
def create_hist_feature(user_ids, recall_dict, click_hist_df, articles_info, article_embeddings, user_embeddings=None, N=1):
    all_user_features = []
    for user_id in tqdm(user_ids):
        # 获取最后N次点击
        hist_N_user_click = click_hist_df[click_hist_df['user_id'] == user_id]['click_article_id'][-N:]
        
        for rank, (article_id, score, label) in enumerate(recall_dict[user_id]):
            curr_article_created_time = articles_info[articles_info['article_id'] == article_id]['created_at_ts'].values[0]
            curr_article_words = articles_info[articles_info['article_id'] == article_id]['words_count'].values[0]

            single_user_feature = [user_id, article_id]

            # 保存相似度
            sim_fea = []
            # 保存创建时间差
            time_fea = []
            # 保存单词数差
            word_fea = []

            for hist_item in hist_N_user_click:
                hist_article_created_time = articles_info[articles_info['article_id'] == hist_item]['created_at_ts'].values[0]
                hist_article_words = articles_info[articles_info['article_id'] == hist_item]['words_count'].values[0]

                sim_fea.append(np.dot(article_embeddings[article_id], article_embeddings[hist_item]))
                time_fea.append(abs(curr_article_created_time - hist_article_created_time))
                word_fea.append(abs(hist_article_words - curr_article_words))

            single_user_feature.extend(sim_fea)
            single_user_feature.extend(time_fea)
            single_user_feature.extend(word_fea)
            # 再添加一些相似性的统计特征
            single_user_feature.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])

            # 如果有user_embedding，那么就计算user和article之间的相似度
            if user_embeddings is not None:
                single_user_feature.extend(np.dot(user_embeddings[user_id], article_embeddings[user_id]))

            # 再将基本特征加入
            single_user_feature.extend([score, rank, label])

            # 加入总特征表中
            all_user_features.append(single_user_feature)
    
    # 将所有的特征表转换为dataframe
    # 定义列名
    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)]
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    satis_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_avg']
    user_item_sim_cols = ['user_item_sim'] if user_embeddings else []
    user_score_rank_labnel_col = ['score', 'rank', 'label']
    all_cols = id_cols + sim_cols + time_cols + word_cols + satis_cols + user_item_sim_cols + user_score_rank_labnel_col

    return pd.DataFrame(all_user_features, columns=all_cols)

# 获取用户使用最多的设备号码
def get_user_device_habit(all_data: pd.DataFrame, cols):
    data = all_data[cols]

    # value_counts后的index是设备编号，值为该设备出现的次数
    user_device_info = data.groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()

    return user_device_info


def get_user_time_habit(all_data: pd.DataFrame, cols):
    data = all_data[cols]
    # 归一化
    data['click_timestamp'] = (data['click_timestamp'] - data['click_timestamp'].min()) / (data['click_timestamp'].max() - data['click_timestamp'].min())
    data['created_at_ts'] = (data['created_at_ts'] - data['created_at_ts'].min()) / (data['created_at_ts'].max() - data['created_at_ts'].min())


    user_time_habit = data.groupby('user_id').agg('mean').reset_index()
    user_time_habit.rename(columns={'click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob1'})
    return user_time_habit

# 获取用户的主题爱好
def get_user_theme_habit(all_data: pd.DataFrame, cols):
    data = all_data[cols]

    # value_counts后的index是设备编号，值为该设备出现的次数
    user_theme_info = data.groupby('user_id').agg({list}).reset_index()

    # user_theme_habit_info = pd.DataFrame()
    # user_theme_habit_info['user_id'] = user_theme_info
    user_theme_info.rename(columns={'category_id': 'cate_list'}, inplace=True)

    return user_theme_info


# 用户阅读文章的字数的偏好
def get_user_words_cnt(all_data: pd.DataFrame, cols):
    data = all_data[cols]
    user_words_info = data.groupby('user_id').agg('mean').reset_index()

    user_words_info.rename(columns={'words_count': 'word_hob'}, inplace=True)
    return user_words_info

# 为特征添加一列，即召回文章的主题是否在用户的主题偏好中
def get_recall_is_in_theme_habit(all_data_feature: pd.DataFrame):
    all_data_feature['is_in_cat_habit'] = all_data_feature.apply(lambda x: 1 if x['category_id'] in x['cate_list'] else 0)
    return all_data_feature

# 用户相关特征
# 用户活跃度
# 如果某个用户点击文章之间的时间间隔比较小， 同时点击的文章次数很多的话， 那么我们认为这种用户一般就是活跃用户, 
# 当然衡量用户活跃度的方式可能多种多样， 这里我们只提供其中一种，我们写一个函数， 得到可以衡量用户活跃度的特征，逻辑如下：
# 1. 首先根据用户user_id分组， 对于每个用户，计算点击文章的次数， 两两点击文章时间间隔的均值
# 2. 把点击次数取倒数和时间间隔的均值统一归一化，然后两者相加合并，该值越小， 说明用户越活跃
# 3. 注意， 上面两两点击文章的时间间隔均值， 会出现如果用户只点击了一次的情况，这时候时间间隔均值那里会出现空值， 对于这种情况最后特征那里给个大数进行区分
# 这个的衡量标准就是先把点击的次数取到数然后归一化， 然后点击的时间差归一化， 然后两者相加进行合并， 该值越小， 说明被点击的次数越多， 且间隔时间短。
def get_user_active_level(all_data: pd.DataFrame, cols):
    data = all_data[cols]
    data = data.sort_values(by=['user_id', 'click_timestamp'])
    user_act = pd.DataFrame(data.groupby('user_id').agg({'click_article_id': np.size, 'click_timestamp': {list}}).values, 
                            columns=['user_id', 'click_size', 'click_timestamp'])
    
    def time_diff_mean(x):
        if len(x) == 1:
            return 1
        else:
            return np.mean(j - i for i, j in zip(x[1:], x[:-1]))
        
    user_act['time_diff_mean'] = user_act.groupby('user_id')['click_timestamp'].apply(lambda x: time_diff_mean(x))
    user_act['click_size'] = 1 / user_act['click_size']

    # 进行归一化
    user_act['click_size'] = (user_act['click_size'] - user_act['click_size'].min()) / (user_act['click_size'].max() - user_act['click_size'].min())
    user_act['time_diff_mean'] = (user_act['time_diff_mean'] - user_act['time_diff_mean'].min()) / (user_act['time_diff_mean'].max() - user_act['time_diff_mean'].min())
    user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']

    del user_act['click_timestamp']

    return user_act

# 文章的热度
# 和上面同样的思路， 如果一篇文章在很短的时间间隔之内被点击了很多次， 说明文章比较热门，实现的逻辑和上面的基本一致， 只不过这里是按照点击的文章进行分组：
# 1. 根据文章进行分组， 对于每篇文章的用户， 计算点击的时间间隔
# 2. 将用户的数量取倒数， 然后用户的数量和时间间隔归一化， 然后相加得到热度特征， 该值越小， 说明被点击的次数越大且时间间隔越短， 文章比较热
def get_article_hot_level(all_data: pd.DataFrame, cols):
    data = all_data[cols]
    data = data.sort_values(by=['click_article_id', 'click_timestamp'])
    item_act = pd.DataFrame(data.groupby('click_article_id').agg({'click_article_id': np.size, 'click_timestamp': {list}}), 
                        columns=['click_article_id', 'user_num', 'click_timestamp'])
    
    def time_diff_mean(x):
        if len(x) == 1:
            return 1
        else:
            return np.mean(j - i for i, j in zip(x[1:], x[:-1]))
        
    item_act['time_diff_mean_item'] = item_act.groupby('click_article_id').apply(lambda x: time_diff_mean(x))
    item_act['user_num'] = 1 / item_act['user_num']

    item_act['user_num'] = (item_act['user_num'] - item_act['user_num'].min()) / (item_act['user_num'].max() - item_act['user_num'].min())
    item_act['time_diff_mean_item'] = (item_act['time_diff_mean_item'] - item_act['time_diff_mean_item'].min()) / (item_act['time_diff_mean_item'].max() - item_act['time_diff_mean_item'].min())
    item_act['hot_level'] = item_act['user_num'] + item_act['time_diff_mean_item']
    
    del item_act['click_timestamp']

    return item_act


##############################################
def get_all_features(data_path: str, recall_list_dict: dict):
    # 获取文章信息
    article_info = get_article_data()
    article_embedding = get_article_embedding_data()
    # 获取recall的df以及recall的dict
    recall_df = recall_list_to_df(recall_list_dict)
    recall_dict = recall_df_to_dict(recall_df)

    # click_hist_df, click_last_df = get_hist_and_last_click(all_click)
    # 获取csv直接读出来的train, val和test数据
    click_train, click_val, test_data, val_ans = get_train_val_test_raw_data(data_path)
    # 获取训练数据的hist和last
    click_train_hist, click_train_last = get_hist_and_last_click(click_train)

    # 获取验证数据的hist和last
    click_val_hist, click_val_last = click_val, val_ans
    # 获取测试数据的hist和last
    click_test_hist, click_test_last = get_hist_and_last_click(test_data)
    
    # 获取训练数据中用户的召回列表
    train_recall_df = recall_df[recall_df['user_id'].isin(click_train_hist['user_id'].unique())]
    # 训练数据加上标签
    train_label_recall_df = set_recall_label(train_recall_df, click_train_last, is_test=False)
    # 进行负采样
    train_label_recall_df = neg_sample_recall_data(train_label_recall_df)

    if click_val is not None:
        val_recall_df = recall_df[recall_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_label_recall_df = set_recall_label(val_recall_df, click_val_last, is_test=False)
        val_label_recall_df = neg_sample_recall_data(val_label_recall_df)
    else:
        val_label_recall_df = None

    test_recall_df = recall_df[recall_df['user_id'].isin(click_test_hist['user_id'].unique())]
    test_label_recall_df = set_recall_label(test_recall_df, click_test_last, is_test=False)



    user_article_raw_data_df = pd.merge(click_train, article_info, on=['click_article_id'], how='left')


    # 获取feature
    train_user_hist_feature = create_hist_feature(train_label_recall_df['user_id'].unique(), recall_dict, 
                                                  click_train_hist, article_info, article_embedding)    
    train_user_words_feature = get_user_time_habit(user_article_raw_data_df[['user_id', 'created_at_ts', 'click_timestamp']])
    train_user_device_feature = get_user_device_habit(user_article_raw_data_df[['user_id', 'click_deviceGroup']])
    train_user_active_feature = get_user_active_level(user_article_raw_data_df[['user_id', 'click_article_id', 'click_timestamp']])
    train_user_theme_feature = get_user_theme_habit(user_article_raw_data_df[['user_id', 'category_id']])
    
    # Merge所有的特征
    user_all_feature = pd.merge(train_user_hist_feature, train_user_words_feature, on='user_id')
    user_all_feature = user_all_feature.merge(user_all_feature, train_user_device_feature, on='user_id')
    user_all_feature = user_all_feature.merge(user_all_feature, train_user_active_feature, on='user_id')
    user_all_feature = user_all_feature.merge(user_all_feature, train_user_theme_feature, on='user_id')
    user_all_feature = user_all_feature.merge(user_all_feature, article_info, on='click_article_id')
    user_all_feature = get_recall_is_in_theme_habit(user_all_feature)
    
    return user_all_feature

if __name__ == '__main__':
    get_all_features()
    