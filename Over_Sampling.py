def func_over_sampling(start=1, end):
    """
    上采样一些原始路况数据

    start: int(defaut=1)
        第一个路况数据编号（取值1-30）
    end: int
        最后一个数据（取值1-30）
    return: 无，会在当前目录下生成一个类别分布均衡的路况数据csv文件
    """
    data = pd.DataFrame()
    for i in range(start, end + 1):
        if i < 10:
            day = '0' + str(i)
        else:
            day = str(i)
        file_name = "201907" + day + '.txt'
        print('start read file {} ...'.format(file_name))
        df = pd.read_csv(file_name, sep=' ', header=None)
        data = pd.concat([data, df])
    
    data_0 = data[data[1] == 1]
    data_1 = data[data[1] == 2]
    data_2 = pd.concat([data[data[1] == 3], data[data[1] == 4]]) # 状态3或4
    print("畅通：{}，缓行：{}，拥挤：{}".format(len(data_0), len(data_1), len(data_2)))

    # 方法一：随机上采样
    index = np.random.choice(range(0, len(data_2)), size=len(data_0), replace=True)
    data_2_over_sampling = data_2.iloc[index]

    index = np.random.choice(range(0, len(data_1)), size=len(data_0), replace=True)
    data_1_over_sampling = data_1.iloc[index]

    data_over_sampling = pd.concat([pd.concat([data_0, data_2_over_sampling]), data_1_over_sampling])
    
    data_over_sampling.to_csv('over_sampling.csv', index=False, encoding='utf8', sep=' ', header=None)

  
