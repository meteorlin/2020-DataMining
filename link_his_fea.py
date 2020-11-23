import pandas as pd
from tqdm import tqdm
from collections import Counter
import os
from numpy import mean
import pickle

def get_statelist(feature1, feature2, feature3, feature4, feature5):
    state = []
    for feature in (feature1, feature2, feature3, feature4, feature5):
        feature = [int(i.split(':')[-1].split(',')[2]) for i in feature.split(' ')]
        state += feature
    
    count0 = state.count(0)
    count1 = state.count(1)
    count2 = state.count(2)
    count3 = state.count(3)
    count4 = state.count(4)

    return [count0, count1, count2, count3, count4]

def get_speedlist(f1, f2, f3, f4, f5):
    speed = []
    for f in (f1, f2, f3, f4, f5):
        f = [float(i.split(':')[-1].split(',')[0]) for i in f.split(' ')]
        speed += f
    aa = [max(speed), min(speed), mean(speed)]
    return aa

def get_etalist(f1, f2, f3, f4, f5):
    eta = []
    for f in (f1, f2, f3, f4, f5):
        f = [float(i.split(':')[-1].split(',')[1]) for i in f.split(' ')]
        eta += f
    aa = [max(eta), min(eta), mean(eta)]
    return aa

def get_cntlist(f1, f2, f3, f4, f5):
    cnt = []
    for f in (f1, f2, f3, f4, f5):
        f = [int(i.split(':')[-1].split(',')[3]) for i in f.split(' ')]
        cnt += f
    aa = [max(cnt), min(cnt), mean(cnt)]
    return aa


def gen_his_fea(path, dic_link_his, mode='is_train'):
    df = pd.read_csv(path, sep=';', header=None, names=['basic', 'now', '-28', '-21', '-14', '-7'])
    df['linkid'] = df['basic'].apply(lambda x: x.split(' ')[0])
    del df['basic']
    
    # 历史state计数及概率
    df['state_list'] = df.apply(lambda x: get_statelist(x['now'], x['-28'], x['-21'], x['-14'], x['-7']), axis=1)
    df['state_0_count'] = df['state_list'].apply(lambda x: x[0])
    df['state_1_count'] = df['state_list'].apply(lambda x: x[1])
    df['state_2_count'] = df['state_list'].apply(lambda x: x[2])
    df['state_3_count'] = df['state_list'].apply(lambda x: x[3])
    df['state_4_count'] = df['state_list'].apply(lambda x: x[4])
    del df['state_list']

    # 历史speed最大最小平均
    df['speed_list'] = df.apply(lambda x: get_speedlist(x['now'], x['-28'], x['-21'], x['-14'], x['-7']), axis=1)
    df['speed_max'] = df['speed_list'].apply(lambda x: x[0])
    df['speed_min'] = df['speed_list'].apply(lambda x: x[1])
    df['speed_mean'] = df['speed_list'].apply(lambda x: x[2])
    del df['speed_list']

    # 历史eta最大最小平均
    df['eta_list'] = df.apply(lambda x: get_etalist(x['now'], x['-28'], x['-21'], x['-14'], x['-7']), axis=1)
    df['eta_max'] = df['eta_list'].apply(lambda x: x[0])
    df['eta_min'] = df['eta_list'].apply(lambda x: x[1])
    df['eta_mean'] = df['eta_list'].apply(lambda x: x[2])
    del df['eta_list']

    # 历史eta最大最小平均
    df['cnt_list'] = df.apply(lambda x: get_cntlist(x['now'], x['-28'], x['-21'], x['-14'], x['-7']), axis=1)
    df['cnt_max'] = df['cnt_list'].apply(lambda x: x[0])
    df['cnt_min'] = df['cnt_list'].apply(lambda x: x[1])
    df['cnt_mean'] = df['cnt_list'].apply(lambda x: x[2])
    del df['cnt_list']
    
    del df['now']
    del df['-28']
    del df['-21']
    del df['-14']
    del df['-7']

    for link, group in df.groupby('linkid'):
        if link in dic_link_his.keys():
            row = dic_link_his[link]
            row['0_count'] += group['state_0_count'].sum()
            row['1_count'] += group['state_1_count'].sum()
            row['2_count'] += group['state_2_count'].sum()
            row['3_count'] += group['state_3_count'].sum()
            row['4_count'] += group['state_4_count'].sum()
            row['avg_speed'] = (group['speed_mean'].mean() + row['avg_speed']) / 2
            row['max_speed'] = max(group['speed_max'].max(), row['max_speed'])
            row['min_speed'] = min(group['speed_min'].min(), row['min_speed'])
            row['avg_eta'] = (group['eta_mean'].mean() + row['avg_eta']) / 2
            row['max_eta'] = max(group['eta_max'].max(), row['max_eta'])
            row['min_eta'] = min(group['eta_min'].min(), row['min_eta'])
            row['avg_cnt'] = (group['cnt_mean'].mean() + row['avg_cnt']) / 2
            row['max_cnt'] = max(group['cnt_max'].max(), row['max_cnt'])
            row['min_cnt'] = min(group['cnt_min'].min(), row['min_cnt'])
        else:
            new = {
                '0_count': group['state_0_count'].sum(),
                '1_count': group['state_1_count'].sum(),
                '2_count': group['state_2_count'].sum(),
                '3_count': group['state_3_count'].sum(),
                '4_count': group['state_4_count'].sum(),
                'avg_speed': group['speed_mean'].mean(),
                'max_speed': group['speed_max'].max(),
                'min_speed': group['speed_min'].min(),
                'avg_eta': group['eta_mean'].mean(),
                'max_eta': group['eta_max'].max(),
                'min_eta': group['eta_min'].min(),
                'avg_cnt': group['cnt_mean'].mean(),
                'max_cnt': group['cnt_max'].max(),
                'min_cnt': group['cnt_min'].min()
            }
            dic_link_his[link] = new
    return dic_link_his


'''
dic_link_his = {
    linkid: {
                0_count: n, 
                1_count: n, 
                2_count: n,
                3_count: n, 
                4_count: n, 
                avg_speed: n, 
                max_speed: n, 
                min_speed: n,
                avg_eta: n,  
                max_eta: n,
                min_eta:n,
                avg_cnt: n,
                max_cnt: n,
                min_cnt n:
            }

}
'''
if __name__ == '__main__':
    dic_link_his = {}
    # 读取存放数据集的文件夹
    rootdir = '/data/cty/CCF_BDCI/数据集/traffic'
    lst = os.listdir(rootdir)              # 列出文件夹下所有的目录与文件
    for i in tqdm(range(len(lst))):
        path = os.path.join(rootdir, lst[i])
        if os.path.isfile(path):
            dic_link_his =  gen_his_fea(path, dic_link_his)

    f = open('/data/cty/CCF_BDCI/his/link_his_dic.pkl', 'wb')
    pickle.dump(dic_link_his, f, pickle.HIGHEST_PROTOCOL)
    f.close()
