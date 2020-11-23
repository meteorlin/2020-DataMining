import pandas as pd
from tqdm import tqdm
from collections import Counter
import os
from numpy import mean
import pickle
        

def get_dict(dic, data):
    for xx in tqdm(range(len(data))):
        row = data[xx]
        linkid = row[0]
        label = row[1]
        cur = row[2]
        pred = row[3]
        # 统计label
        if (linkid, pred) in dic.keys():
            dic[(linkid, pred)].append(label)
        else:
            dic[(linkid, pred)] = [label]

        # 统计cur
        for i in range(5):
            time = cur - i
            if (linkid, time) in dic.keys():
                dic[(linkid, time)].append(row[8 - i])
            else:
                dic[(linkid, time)] = [row[8 - i]]
        # 统计his
        for i in range(5):
            time = pred + i
            if (linkid, time) in dic.keys():
                dic[(linkid, time)].append(row[9 + i])
                dic[(linkid, time)].append(row[14 + i])
                dic[(linkid, time)].append(row[19 + i])
                dic[(linkid, time)].append(row[24 + i])
            else:
                dic[(linkid, time)] = [row[9 + i]]
                dic[(linkid, time)].append(row[14 + i])
                dic[(linkid, time)].append(row[19 + i])
                dic[(linkid, time)].append(row[24 + i])

    return dic


def gen_table(path, dic):
    df = pd.read_csv(path, sep=';', header=None,
                     names=['basic', 'now', '28', '21', '14', '7'])
    df['linkid'] = df['basic'].apply(lambda x: x.split(' ')[0])
    df['label'] = df['basic'].apply(lambda x: int(x.split(' ')[1]))
    df['cur_slice'] = df['basic'].apply(lambda x: int(x.split(' ')[2]))
    df['pred_slice'] = df['basic'].apply(lambda x: int(x.split(' ')[3]))
    del df['basic']

    for i in range(5):
        df[f'state_cur_{i}'] = df.apply(lambda x: int(x['now'].split(' ')[i].split(':')[-1].split(',')[2]), axis=1)
    del df['now']

    for time in ('28', '21', '14', '7'):
        for j in range(5):
            df[f'state_{time}_{j}'] = df.apply(lambda x: int(x[time].split(' ')[j].split(':')[-1].split(',')[2]), axis=1)
        del df[time]

    data_lst = df.values.tolist()
    dic = get_dict(dic, data_lst)
    
    return dic


if __name__ == '__main__':
    link_time_table = {}
    rootdir = '/data/cty/CCF_BDCI/数据集/traffic'
    lst = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in tqdm(range(len(lst))):
        path = os.path.join(rootdir, lst[i])
        if os.path.isfile(path):
            link_time_table = gen_table(path, link_time_table)

    for key in link_time_table.keys():
        cnt = len(link_time_table[key])
        _0_count = Counter(link_time_table[key])[0]
        _1_count = Counter(link_time_table[key])[1]
        _2_count = Counter(link_time_table[key])[2]
        _3_count = Counter(link_time_table[key])[3]
        _4_count = Counter(link_time_table[key])[4]
        link_time_table[key] = [cnt, _0_count/cnt, _1_count/cnt, _2_count/cnt, _3_count/cnt, _4_count/cnt]
    f = open('/data/cty/CCF_BDCI/his/link_time_table.pkl', 'wb')
    pickle.dump(link_time_table, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    df = pd.DataFrame(link_time_table).T    
    df.columns = ['cnt','0_prob','1_prob','2_prob','3_prob','4_prob']
    df['cnt'] = df['cnt'].astype('int')
    df.to_csv('/data/cty/CCF_BDCI/his/link_time_table.csv')