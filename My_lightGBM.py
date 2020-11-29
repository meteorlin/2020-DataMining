import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import EasyEnsembleClassifier

warnings.filterwarnings("ignore")


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    scores = f1_score(y_true=labels, y_pred=np.argmax(preds.reshape(3, -1), axis=0), average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    return 'f1_score', scores, True


# 样本权重设置
def set_weight(x, a, b, c):
    if x == 2:
        return c
    elif x == 1:
        return b
    else:
        return a


def lgb_train(train_: pd.DataFrame, valid_: pd.DataFrame, test_: pd.DataFrame, use_train_feats: list, id_col: str,
              label: str,
              a: float, b: float, c: float, use_cart=False, cate_cols=None) -> pd.DataFrame:
    if not cate_cols:
        cate_cols = []
    print('data shape:\ntrain--{}\nvali--{}\ntest--{}'.format(train_.shape, valid_.shape, test_.shape))
    print('Use {} features ...'.format(len(use_train_feats)))
    print('Use lightgbm to train ...')

    n_class = train_[label].nunique()  # 返回不同值的个数
    importance_df = pd.DataFrame()
    importance_df["Feature"] = use_train_feats

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

    train_x = train_.loc[train_.index, use_train_feats]
    train_y = train_.loc[train_.index, label]
    valid_x = valid_.loc[valid_.index, use_train_feats]
    valid_y = valid_.loc[valid_.index, label]

    if use_cart:
        df_train = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)  # 适应非DataFrame数据
        df_valid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
    else:
        df_train = lgb.Dataset(train_x, label=train_y,
                               weight=train_y.apply(set_weight, a=a, b=b, c=c))
        df_valid = lgb.Dataset(valid_x, label=valid_y,
                               weight=valid_y.apply(set_weight, a=a, b=b, c=c))  # 验证集也设置样本权重？

    clf = lgb.train(
        params=params,
        train_set=df_train,
        num_boost_round=10000,
        valid_sets=[df_valid],
        valid_names=['valid'],
        early_stopping_rounds=100,
        verbose_eval=10,
        feval=f1_score_eval
    )

    test_pred = clf.predict(test_[use_train_feats], num_iteration=clf.best_iteration)
    clf.save_model(f'./save_models/model')
    # print(test_pred, type(test_pred))
    test_[label] = np.argmax(test_pred, axis=1) + 1
    importance_df[f'imp'] = clf.feature_importance(importance_type='gain')
    importance_df.sort_values(by='imp', ascending=False, inplace=True)
    print(importance_df[['Feature', 'imp']].head(20))
    valid_pred = np.argmax(clf.predict(valid_x, num_iteration=clf.best_iteration), axis=1)
    print(classification_report(valid_y, valid_pred, digits=4))
    test_['future_slice_id'] = test_['future_slice_id_substitute']  # 恢复
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

    attr = pd.read_csv('attribute.txt', sep='\t',
                       names=['link', 'length', 'direction', 'path_class', 'speed_class', 'LaneNum', 'speed_limit',
                              'level', 'width'], header=None)  # 道路属性
    link_static_profiling = pd.read_csv("link_his_fea_no_neighbor.csv", sep=',')  # link静态画像信息
    word2vec_info = pd.read_csv("word2vec_test_3.csv", header=None)  # DeepWalk
    word2vec_info['link'] = word2vec_info[0].apply(lambda x: x.split(" ")[0])
    word2vec_info['link'] = word2vec_info['link'].apply(int)  # 类型转换
    for i in range(100):
        col_name = 'vec' + str(i)
        word2vec_info[col_name] = word2vec_info[0].apply(lambda x: float(x.split(" ")[i + 1]))
        # word2vec_info[col_name] = word2vec_info[col_name].apply(float)  # 类型转换
    del word2vec_info[0]
    print(word2vec_info)

    link_sum = pd.read_csv("link_time_table_10.csv")
    link_sum.columns = ['link', 'future_slice_id', 'cnt', '0_prob', '1_prob', '2_prob', '3_prob', '4_prob']
    # link_sum.drop(index=(len(link_sum) - 1), inplace=True)  # 删除最后一行NAN值
    link_sum['link'] = link_sum['link'].astype(int)
    link_sum['future_slice_id'] = link_sum['future_slice_id'].astype(int)
    print("---1---\n", link_sum['link'][:5], link_sum['future_slice_id'][:5])
    print(link_sum.columns)

    link_up_sum = pd.read_csv("df_sum_jupyter_up_tol_new1.csv")
    link_up_sum.drop(index=(len(link_up_sum) - 1), inplace=True)  # 删除最后一行NAN值
    link_up_sum['link'] = link_up_sum['link'].astype(int)
    link_up_sum['future_slice_id'] = link_up_sum['future_slice_id'].astype(int)
    print("---2---\n", link_up_sum['link'][:5], link_up_sum['future_slice_id'][:5])
    print(link_up_sum.columns)

    link_down_sum = pd.read_csv("df_sum_jupyter_tol_down_new1.csv")
    link_down_sum.drop(index=(len(link_down_sum) - 1), inplace=True)  # 删除最后一行NAN值
    link_down_sum['link'] = link_down_sum['link'].astype(int)
    link_down_sum['future_slice_id'] = link_down_sum['future_slice_id'].astype(int)
    print("---3---\n", link_down_sum['link'][:5], link_down_sum['future_slice_id'][:5])
    print(link_down_sum.columns)

    # train = get_train_data(1)
    # # train = train.sample(frac=1.0)  # 数据打乱！
    # # print(Counter(train['label']))
    # valid = pd.read_csv("valid_set.csv")  # 验证集（仅供评估模型性能，不会直接影响训练过程，但可以通过人为“调参”间接影响训练过程）
    # test = pd.read_csv('is_test.csv')  # 线上测试集

    train = pd.read_csv("train_set_20190704.csv")
    test = pd.read_csv("test_set.csv")
    valid = pd.read_csv("valid_set_lstm.csv")

    train['future_slice_id_substitute'] = train['future_slice_id']  # 另存
    train['future_slice_id'] = train['future_slice_id'].apply(lambda x: x / 10).astype(int)  # 桶
    test['future_slice_id_substitute'] = test['future_slice_id']  # 另存
    test['future_slice_id'] = test['future_slice_id'].apply(lambda x: int(x / 10))  # 桶
    valid['future_slice_id_substitute'] = valid['future_slice_id']  # 另存
    valid['future_slice_id'] = valid['future_slice_id'].apply(lambda x: int(x / 10))  # 桶

    train = train.merge(attr, on='link', how='left')
    test = test.merge(attr, on='link', how='left')
    valid = valid.merge(attr, on='link', how='left')

    train = train.merge(link_static_profiling, on='link', how='left')
    test = test.merge(link_static_profiling, on='link', how='left')
    valid = valid.merge(link_static_profiling, on='link', how='left')

    train = train.merge(word2vec_info, on='link', how='left')
    test = test.merge(word2vec_info, on='link', how='left')
    valid = valid.merge(word2vec_info, on='link', how='left')

    train = train.merge(link_sum, on=['link', 'future_slice_id'], how='left')
    test = test.merge(link_sum, on=['link', 'future_slice_id'], how='left')
    valid = valid.merge(link_sum, on=['link', 'future_slice_id'], how='left')

    train = train.merge(link_up_sum, on=['link', 'future_slice_id'], how='left')
    test = test.merge(link_up_sum, on=['link', 'future_slice_id'], how='left')
    valid = valid.merge(link_up_sum, on=['link', 'future_slice_id'], how='left')

    train = train.merge(link_down_sum, on=['link', 'future_slice_id'], how='left')
    test = test.merge(link_down_sum, on=['link', 'future_slice_id'], how='left')
    valid = valid.merge(link_down_sum, on=['link', 'future_slice_id'], how='left')

    train_nan_col = train.isnull().sum(axis=0)
    test_nan_col = test.isnull().sum(axis=0)
    train_nan_col.to_csv("train_nan_col_partial_lstm.csv", index=False, encoding='utf8')
    test_nan_col.to_csv("test_nan_col_partial_lstm.csv", index=False, encoding='utf8')
    pd.DataFrame(train.columns).to_csv("Features_partial_lstm.csv", index=False, encoding='utf-8')

    # NAN值填充
    train = train.fillna(value=0)
    valid = valid.fillna(value=0)
    test = test.fillna(value=0)

    use_cols = [i for i in train.columns if i not in ['link', 'label', 'current_slice_id']]

    # 过采样SMOTE：link不作为特征之一
    # model_smote = SMOTE()  # 建立smote模型对象
    # x, y = train.loc[
    #            train.index, use_cols], \
    #        train.loc[train.index, 'label']
    # x_smote_sample, y_smote_sample = model_smote.fit_sample(x, y)
    # x_smote_sample = pd.DataFrame(x_smote_sample, columns=use_cols)
    # y_smote_sample = pd.DataFrame(y_smote_sample, columns=['label'])
    # smote_sample = pd.concat([x_smote_sample, y_smote_sample], axis=1)
    # print(smote_sample)
    # groupBy_data_smote = smote_sample.groupby('label').count()
    # print(groupBy_data_smote)
    # train = smote_sample
    # print(train)

    # 下采样NearMiss
    # near_miss = NearMiss(version=1)
    # x, y = train.loc[
    #            train.index, use_cols], \
    #        train.loc[train.index, 'label']
    # x_near_miss_sample, y_near_miss_sample = near_miss.fit_sample(x, y)
    # print(x_near_miss_sample, y_near_miss_sample)
    # train = pd.concat([x_near_miss_sample, y_near_miss_sample], axis=1)
    # print(Counter(train))  # 类别统计
    # print(train)
    # train = train.sample(frac=1.0)  # 数据打乱！
    # print(train)

    # 下采样EasyEnsemble
    # model_EasyEnsemble = EasyEnsembleClassifier()  # Adaboost算法
    # x, y = train.loc[
    #                train.index, use_cols].values, \
    #            train.loc[train.index, 'label'].values
    # model_EasyEnsemble.fit(x, y)
    # test_x = test.loc[test.index, use_cols].values
    # # print(test_x[np.isinf(test_x).T.any()])  # inf
    # # print(test_x[np.isnan(test_x).T.any()])  # NAN
    # # test_x = test_x.replace(np.inf, 0)  # 无穷值替换
    # test_y = model_EasyEnsemble.predict(test_x)
    # print(test_y)
    # test['label'] = test_y
    # result = test[['link', 'current_slice_id', 'future_slice_id', 'label']]
    # result.to_csv('Adaboost_model_result.csv', index=False, encoding='UTF-8')

    sub = lgb_train(train, valid, test, use_cols, 'link', 'label', 1, 2.5, 4)

    sub.to_csv('public_baseline_plus_lstm.csv', index=False, encoding='utf8')

