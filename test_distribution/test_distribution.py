from collections import Counter

import numpy as np
import pandas as pd
import gc
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import lightgbm as lgb

def gen_feats(txt_path, mode='is_train'):
    df_ = pd.read_csv(txt_path, sep=';', header=None)
    df_['link'] = df_[0].apply(lambda x: x.split(' ')[0])
    if mode == 'is_train':
        df_['label'] = df_[0].apply(lambda x: int(x.split(' ')[1]))
        df_['label'] = df_['label'].apply(lambda x: 3 if x > 3 else x)
        df_['label'] -= 1
        df_['current_slice_id'] = df_[0].apply(lambda x: int(x.split(' ')[2]))
        df_['future_slice_id'] = df_[0].apply(lambda x: int(x.split(' ')[3]))
    else:
        df_['label'] = -1
        df_['current_slice_id'] = df_[0].apply(lambda x: int(x.split(' ')[2]))
        df_['future_slice_id'] = df_[0].apply(lambda x: int(x.split(' ')[3]))

    df_.drop([0], axis=1, inplace=True)
    df_['time_diff'] = df_['future_slice_id'] - df_['current_slice_id']

    for ii in range(0, 5):
        df_['curr_state'] = df_[1].apply(lambda x: x.split(' ')[ii].split(':')[-1])
        if ii == 4:
            flg = 'curr'
        else:
            flg = f'rec_{(4 - ii) * 2}'
        df_[f'{flg}_speed'] = df_['curr_state'].apply(lambda x: x.split(',')[0])
        df_[f'{flg}_eta'] = df_['curr_state'].apply(lambda x: x.split(',')[1])
        df_[f'{flg}_cnt'] = df_['curr_state'].apply(lambda x: x.split(',')[3])
        df_[f'{flg}_state'] = df_['curr_state'].apply(lambda x: x.split(',')[2])
    df_.drop([1], axis=1, inplace=True)
    print('recent_gen complete')

    for ii in range(2, 6):
        df_['his_info'] = df_[ii].apply(lambda x: [j.split(':')[-1] for j in x.split(' ')])
        flg = f'his_{(6 - ii) * 7}'
        df_['his_speed'] = df_['his_info'].apply(lambda x: np.array([j.split(',')[0] for j in x], dtype='float16'))
        df_[f'{flg}_speed_min'] = df_['his_speed'].apply(lambda x: x.min())
        df_[f'{flg}_speed_max'] = df_['his_speed'].apply(lambda x: x.max())
        df_[f'{flg}_speed_mean'] = df_['his_speed'].apply(lambda x: x.mean())
        df_[f'{flg}_speed_std'] = df_['his_speed'].apply(lambda x: x.std())

        df_['his_eta'] = df_['his_info'].apply(lambda x: np.array([j.split(',')[1] for j in x], dtype='float16'))
        df_[f'{flg}_eta_min'] = df_['his_eta'].apply(lambda x: x.min())
        df_[f'{flg}_eta_max'] = df_['his_eta'].apply(lambda x: x.max())
        df_[f'{flg}_eta_mean'] = df_['his_eta'].apply(lambda x: x.mean())
        df_[f'{flg}_eta_std'] = df_['his_eta'].apply(lambda x: x.std())

        df_['his_cnt'] = df_['his_info'].apply(lambda x: np.array([j.split(',')[3] for j in x], dtype='int16'))
        df_[f'{flg}_cnt_min'] = df_['his_cnt'].apply(lambda x: x.min())
        df_[f'{flg}_cnt_max'] = df_['his_cnt'].apply(lambda x: x.max())
        df_[f'{flg}_cnt_mean'] = df_['his_cnt'].apply(lambda x: x.mean())
        df_[f'{flg}_cnt_std'] = df_['his_cnt'].apply(lambda x: x.std())

        df_['his_state'] = df_['his_info'].apply(lambda x: [int(j.split(',')[2]) for j in x])
        df_[f'{flg}_state'] = df_['his_state'].apply(lambda x: Counter(x).most_common()[0][0])
        df_.drop([ii, 'his_info', 'his_speed', 'his_eta', 'his_cnt', 'his_state'], axis=1, inplace=True)
    print('history_gen complete')
    if mode == 'is_train':
        save_path = f"{path}feature/{mode}_{data_date}.csv"
        df_.to_csv(save_path, index=False)
        return save_path
    else:
        save_path = f"{path}feature/is_test.csv"
        df_.to_csv(save_path, index=False)
        return save_path

NFold = 5
data_date = 20190723  # 根据需要进行分布测试的训练集数据的日期更改
path = './'  # 路径是存放生成好的训练集和测试集基础特征的路径

if os.path.exists(f'{path}is_train_{data_date}.csv'):
    train_fea = f'{path}is_train_{data_date}.csv'
else:
    train_fea = gen_feats(f"traffic/{data_date}.txt")
print('train_fea already')
if os.path.exists(f'{path}is_test.csv'):
    test_fea = f'{path}is_test.csv'
else:
    test_fea = gen_feats(f"test.txt", mode='is_test')
print('test_fea already')


train = pd.read_csv(train_fea)
test = pd.read_csv(test_fea)

train['id'] = range(0, train.shape[0])
test['id'] = -1
train['target'] = 1
test['target'] = 0

n_train = train.shape[0]
df = pd.concat([train, test], axis=0)
del train, test
gc.collect()

predictors = list(df.columns.difference(['id', 'target', 'label']))
df_train = df.iloc[:n_train].copy()
cols_to_remove = [c for c in predictors if df_train[c].nunique() == 1]
df.drop(cols_to_remove, axis=1, inplace=True)
predictors = list(df.columns.difference(['id', 'target', 'label'] + cols_to_remove))
print(predictors)

# shuffle dataset
df = df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True, inplace=True)

target = 'target'

lgb_params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'nthread': 4,
    'verbose': -1
}

folds = KFold(n_splits=NFold, shuffle=True, random_state=0)
fold = folds.split(df)
eval_score = 0
n_estimators = 0
eval_preds = np.zeros(df.shape[0])

fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = predictors

for i, (train_idx, test_idx) in enumerate(fold, start=1):
    print("\n[{}] Fold {} of {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, NFold))
    train_x, valid_x = df[predictors].values[train_idx], df[predictors].values[test_idx]
    train_y, valid_y = df[target].values[train_idx], df[target].values[test_idx]
    dtrain = lgb.Dataset(train_x, label=train_y, feature_name=list(predictors))
    dvalid = lgb.Dataset(valid_x, label=valid_y, feature_name=list(predictors))
    eval_results = {}
    bst = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        evals_result=eval_results,
        num_boost_round=5500,
        early_stopping_rounds=100,
        verbose_eval=100
    )

    print("\nRounds: ", bst.best_iteration)
    print("AUC: ", eval_results['valid']['auc'][bst.best_iteration - 1])

    fold_importance_df[f'fold_{i}_imp'] = bst.feature_importance(importance_type='gain')
    n_estimators += bst.best_iteration
    eval_score += eval_results['valid']['auc'][bst.best_iteration - 1]
    eval_preds[test_idx] += bst.predict(valid_x, num_iteration=bst.best_iteration)

n_estimators = int(round(n_estimators / NFold, 0))
eval_score = round(eval_score / NFold, 6)

print("\nModel Report")
print("Rounds: ", n_estimators)
print("AUC: ", eval_score)

five_folds = [f'fold_{f}_imp' for f in range(1, NFold + 1)]
fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
print(fold_importance_df[['Feature', 'avg_imp']].head(20))

df_av = df[['id', 'target']].copy()
df_av['preds'] = eval_preds
df_av_train = df_av[df_av['target'] == 1]
df_av_train = df_av_train.sort_values(by=['preds']).reset_index(drop=True)

df_av_train.preds.plot()
plt.show()

df_av_train[['id', 'preds']].reset_index(drop=True).to_csv(f"adversarial_validation_{data_date}.csv", index=False)
df_av_train.head(20)
