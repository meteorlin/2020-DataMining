import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def get_base_info(x):
    return [i.split(':')[-1] for i in x.split(' ')]


def get_speed(x):
    return np.array([i.split(',')[0] for i in x], dtype='float16')


def get_eta(x):
    return np.array([i.split(',')[1] for i in x], dtype='float16')


def get_state(x):
    return [int(i.split(',')[2]) for i in x]


def get_cnt(x):
    return np.array([i.split(',')[3] for i in x], dtype='int16')


def gen_feats(path, mode='is_train'):
    df = pd.read_csv(path, sep=';', header=None)
    df['link'] = df[0].apply(lambda x: x.split(' ')[0])
    if mode == 'is_train':
        df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
        df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
        df['label'] -= 1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
    else:
        df['label'] = -1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))

    df['time_diff'] = df['future_slice_id'] - df['current_slice_id']

    df['curr_state'] = df[1].apply(lambda x: x.split(' ')[-1].split(':')[-1])
    df['curr_speed'] = df['curr_state'].apply(lambda x: x.split(',')[0])
    df['curr_eta'] = df['curr_state'].apply(lambda x: x.split(',')[1])
    df['curr_cnt'] = df['curr_state'].apply(lambda x: x.split(',')[3])
    df['curr_state'] = df['curr_state'].apply(lambda x: x.split(',')[2])
    del df[0]

    for i in tqdm(range(1, 6)):
        df['his_info'] = df[i].apply(get_base_info)
        if i == 1:
            flg = 'current'
        else:
            flg = f'his_{(6 - i) * 7}'
        df['his_speed'] = df['his_info'].apply(get_speed)
        df[f'{flg}_speed_min'] = df['his_speed'].apply(lambda x: x.min())
        df[f'{flg}_speed_max'] = df['his_speed'].apply(lambda x: x.max())
        df[f'{flg}_speed_mean'] = df['his_speed'].apply(lambda x: x.mean())
        df[f'{flg}_speed_std'] = df['his_speed'].apply(lambda x: x.std())

        df['his_eta'] = df['his_info'].apply(get_eta)
        df[f'{flg}_eta_min'] = df['his_eta'].apply(lambda x: x.min())
        df[f'{flg}_eta_max'] = df['his_eta'].apply(lambda x: x.max())
        df[f'{flg}_eta_mean'] = df['his_eta'].apply(lambda x: x.mean())
        df[f'{flg}_eta_std'] = df['his_eta'].apply(lambda x: x.std())

        df['his_cnt'] = df['his_info'].apply(get_cnt)
        df[f'{flg}_cnt_min'] = df['his_cnt'].apply(lambda x: x.min())
        df[f'{flg}_cnt_max'] = df['his_cnt'].apply(lambda x: x.max())
        df[f'{flg}_cnt_mean'] = df['his_cnt'].apply(lambda x: x.mean())
        df[f'{flg}_cnt_std'] = df['his_cnt'].apply(lambda x: x.std())

        df['his_state'] = df['his_info'].apply(get_state)
        df[f'{flg}_state'] = df['his_state'].apply(lambda x: Counter(x).most_common()[0][0])
        df.drop([i, 'his_info', 'his_speed', 'his_eta', 'his_cnt', 'his_state'], axis=1, inplace=True)
    if mode == 'is_train':
        df.to_csv(f"{mode}_{path.split('/')[-1]}", index=False)
    else:
        df.to_csv(f"is_test.csv", index=False)


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average=None)
    scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
    return 'f1_score', scores, True


# 样本权重设置
def set_weight(x, a, b, c):
    if x == 2:
        return a
    elif x == 1:
        return b
    else:
        return c


def lgb_train(train_: pd.DataFrame, test_: pd.DataFrame, use_train_feats: list, id_col: str, label: str,
              n_splits: int, split_rs: int, a: float, b: float, c: float, is_shuffle=True, use_cart=False, cate_cols=None) -> pd.DataFrame:
    if not cate_cols:
        cate_cols = []
    print('data shape:\ntrain--{}\ntest--{}'.format(train_.shape, test_.shape))
    print('Use {} features ...'.format(len(use_train_feats)))
    print('Use lightgbm to train ...')

    n_class = train_[label].nunique()  # 返回不同值的个数
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_user_id = train_[id_col].unique()  # 返回所有不同的值

    train_[f'{label}_pred'] = 0
    test_pred = np.zeros((test_.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = use_train_feats

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    # print(f"The folds.n_splits is {folds.n_splits}")
    # print(f"The label is {label}")

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):  # 交叉验证划分的是'link'，而不是直接划分数据
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_.loc[train_[id_col].isin(train_user_id[train_idx]), use_train_feats], train_.loc[
            train_[id_col].isin(train_user_id[train_idx]), label]
        valid_x, valid_y = train_.loc[train_[id_col].isin(train_user_id[valid_idx]), use_train_feats], train_.loc[
            train_[id_col].isin(train_user_id[valid_idx]), label]
        print(f'for train user:{len(train_idx)}\nfor valid user:{len(valid_idx)}')

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)  # 适应非DataFrame数据
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y,
                                 weight=train_y.apply(set_weight, a=a, b=b, c=c))
            dvalid = lgb.Dataset(valid_x, label=valid_y,
                                 weight=valid_y.apply(set_weight, a=a, b=b, c=c))  # 验证集也设置样本权重？

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=['valid'],
            early_stopping_rounds=100,  # "valid_0": 第一个验证集
            verbose_eval=100,
            feval=f1_score_eval
        )
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance(importance_type='gain')
        train_.loc[train_[id_col].isin(train_user_id[valid_idx]), f'{label}_pred'] = np.argmax(
            clf.predict(valid_x, num_iteration=clf.best_iteration),
            axis=1)  # 高级操作（‘num_iteration’: Total number of iterations used in the prediction.）
        test_pred += clf.predict(test_[use_train_feats], num_iteration=clf.best_iteration) / folds.n_splits  # 均值

    report = f1_score(train_[label], train_[f'{label}_pred'], average=None)
    print(classification_report(train_[label], train_[f'{label}_pred'], digits=4))  # 'support': 类别样本数
    print('F1_Score: ', report[0] * 0.2 + report[1] * 0.2 + report[2] * 0.6)
    test_[f'{label}_pred'] = np.argmax(test_pred, axis=1)
    test_[label] = np.argmax(test_pred, axis=1) + 1
    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[['Feature', 'avg_imp']].head(20))
    return test_[[id_col, 'current_slice_id', 'future_slice_id', label]]


def get_train_data(num, start=None, end=None):
    if start is None:
        start = 1
    if end is None:
        end = num + start

    data = pd.DataFrame()
    for i in range(start, end):
        if i < 10:
            day = '0' + str(i)
        else:
            day = str(i)
        file_name = "is_train_201907" + day + '.txt'
        print(f"start import {file_name} ...")
        df = pd.read_csv(file_name)
        data = pd.concat([data, df])  # 行拼接
    return data


if __name__ == "__main__":

    # train_path = 'traffic_20190701.txt'
    # test_path = 'test.txt'
    # gen_feats(train_path, mode='is_train')
    # gen_feats(test_path, mode='is_test')

    attr = pd.read_csv('attribute.txt', sep='\t',
                       names=['link', 'length', 'direction', 'path_class', 'speed_class', 'LaneNum', 'speed_limit',
                              'level', 'width'], header=None)  # 道路属性特征

    train = get_train_data(1, 7)  # 提取基本特征后的路况数据（使用gen_feats()函数）
    # train = pd.read_csv("is_train_over_mix_under_sampling_1.csv")  # 重采样后的路况数据
    test = pd.read_csv('is_test.csv')
    train = train.merge(attr, on='link', how='left')
    test = test.merge(attr, on='link', how='left')

    use_cols = [i for i in train.columns if i not in ['link', 'label', 'current_slice_id', 'label_pred']]

    sub = lgb_train(train, test, use_cols, 'link', 'label', 5, 2020, 4, 2.5, 1)

    sub.to_csv('public_baseline.csv', index=False, encoding='utf8')
