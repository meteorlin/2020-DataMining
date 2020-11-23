import pickle
import pandas as pd
from numpy import mean

def main(path):
    f = open(path, 'rb')
    dic = pickle.load(f)
    df = pd.DataFrame(dic)
    df = df.T
    # get prob
    df['0_prob'] = df.apply(lambda x: x['0_count'] / (x['0_count'] + x['1_count'] + x['2_count'] + x['3_count'] + x['4_count']), axis=1)
    df['1_prob'] = df.apply(lambda x: x['1_count'] / (x['0_count'] + x['1_count'] + x['2_count'] + x['3_count'] + x['4_count']), axis=1)
    df['2_prob'] = df.apply(lambda x: x['2_count'] / (x['0_count'] + x['1_count'] + x['2_count'] + x['3_count'] + x['4_count']), axis=1)
    df['3_prob'] = df.apply(lambda x: x['3_count'] / (x['0_count'] + x['1_count'] + x['2_count'] + x['3_count'] + x['4_count']), axis=1)
    df['4_prob'] = df.apply(lambda x: x['4_count'] / (x['0_count'] + x['1_count'] + x['2_count'] + x['3_count'] + x['4_count']), axis=1)
    df.to_csv('/data/cty/CCF_BDCI/his/link_his_fea_no_neighbor.csv')
    

if __name__ == '__main__':
    path = '/data/cty/CCF_BDCI/his/link_his_dic.pkl'
    # main(path)
