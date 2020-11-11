import os
import re
import pandas as pd
import math
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import codecs
import json


def get_file_list(files_dir: str) -> list:
    """
    * 作用：获取当前目录下的所有文件的文件名列表
    :param files_dir: 指定目录
    :return: 指定目录下文件的文件名列表
    """
    return os.listdir(files_dir)


def format_with_zero(num: int) -> str:
    """
    * 作用：格式化输出数字，如1~9输出为01等，为了匹配日期
    :param num: 目标数字
    :return: 格式化数字的字符串
    """
    return str(num) if num > 9 else str("0" + str(num))


def combine_label_index(label: str) -> str:
    """
    * 作用：整合label中的3和4
    :param label:
    :return:
    """
    return label if int(label) <= 3 else 3


# 列表排序list.sort(key=function))中的指定键值排序函数function
def take_slice_id(elem):
    return int(elem["slice_id"])


def take_link_id(elem):
    return int(elem["link"])


def transform_file_to_csv(original_data_dir: str, filename: str,
                          output_path: str = './traffic_to_csv_output', meta_and_recent: bool = True):
    """
    * 作用：将traffic数据分块并转换为csv的单步函数
    * 使用方法：（需要配合文件列表获取方法使用）
    ```
    file_list = get_file_list(args.original_data_dir)
    for item in file_list:
        transform_file_to_csv(original_data_dir, output_path, item)
    ```
    :param original_data_dir: 原始的30个traffic.txt所在的目录
    :param filename: 当前文件名（包含后缀）
    :param output_path: 最终各个结果的输出目录
    :param meta_and_recent: 是否获取`meta_data`和`recent_feature`，默认为“获取”
        （之所以会有这个选项是之前测试的时候输出`history_feature`时有错误，为了不重复输出已经输出的正确结果而设立）
    :return 每个日期会输出3个文件，分别为meta_data（即第一个区块的内容），recent_feature（第二区块内容），history_feature（第三区块内容）
   """

    # 如果已经获取了meta data 和 recent feature，可以设置meta_and_recent为False以提高运行速度
    def feature_data(_link: int, _day_record, container_list: list):
        _inner_list = _day_record.split(' ')
        for _item in _inner_list:
            slice_id, attrs = _item.split(':')
            speed_with_light, speed_without_light, _label, car_count = attrs.split(',')
            container_list.append({
                "link": _link,
                "slice_id": slice_id,
                "speed_with_light": speed_with_light,
                "speed_without_light": speed_without_light,
                "label": combine_label_index(_label),
                "car_count": car_count,
            })

    _date, _ = os.path.splitext(filename)
    _date_time = datetime.datetime(year=int(_date[0:4]), month=int(_date[4:6]), day=int(_date[6:]))
    meta_data = list()
    meta_data_headers = ['link', 'label_meta_data', 'current_slice_id', 'feature_slice_id']

    recent_feature = list()
    history_features = list()
    history_feature_length = 0  # 历史时间片的数量
    feature_headers = ["link", "slice_id", "speed_with_light", "speed_without_light", "label", "car_count"]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(original_data_dir, filename), 'r', encoding='utf-8') as f:
        index = 0
        for line in f:
            all_line_info = line.split(';')
            meta_data_line = all_line_info[0]
            link, label_meta_data, current_slice_id, feature_slice_id = meta_data_line.split(' ')
            if meta_and_recent:
                meta_data.append({
                    "link": link,
                    "label": combine_label_index(label_meta_data),
                    "current_slice_id": current_slice_id,
                    "feature_slice_id": feature_slice_id,
                })

                # recent_feature
                feature_data(link, all_line_info[1], recent_feature)

            # history_feature
            history_feature_length = len(all_line_info[2:])
            if len(history_features) == 0:
                for i in range(history_feature_length):
                    history_features.append(list())
            for date_index, day_record in enumerate(all_line_info[2:]):
                feature_data(link, day_record, history_features[date_index])

            index += 1

        # 数据保存
        # meta信息和recent feature
        if meta_and_recent:
            _meta_data = pd.DataFrame(meta_data)
            # meta_data没有重复值无需去重
            _meta_data.to_csv(os.path.join(output_path, _date + '_meta_data.csv'),
                              header=meta_data_headers,
                              index=False, mode='w', encoding='utf-8')
            del _meta_data
            del meta_data

            recent_feature.sort(key=take_slice_id)  # 按照slice_id进行排序
            _recent_feature = pd.DataFrame(recent_feature)
            _recent_feature = _recent_feature.drop_duplicates()
            _recent_feature.to_csv(os.path.join(output_path, _date + '_recent_feature.csv'),
                                   header=feature_headers,
                                   index=False, mode='w', encoding='utf-8')
            del _recent_feature
            del recent_feature

        # history feature
        for date_index in range(history_feature_length):
            history_features[date_index].sort(key=take_slice_id)  # 按照slice_id进行排序
            _history_feature = pd.DataFrame(history_features[date_index])
            # 去重
            _history_feature = _history_feature.drop_duplicates()
            # 历史时间片是以前28天、前21天、前14天、前7天的顺序排序的
            _history_date = _date_time - datetime.timedelta(days=7 * (4 - date_index))
            _history_date_str = "{year}{month}{day}".format_map({
                "year": _history_date.year,
                "month": format_with_zero(_history_date.month),
                "day": format_with_zero(_history_date.day)
            })
            _history_feature.to_csv(
                os.path.join(output_path, _history_date_str + '_' + _date + '_history_feature.csv'),
                header=feature_headers,
                index=False, mode='w', encoding='utf-8')


def transform_attr_speed_limit_unit(attr_txt_path: str, output_path: str = './'):
    """
    * 作用：将attr.csv中link限速的单位换算：从m/s转换为km/h
    :param attr_txt_path: attr.csv文件所在的路径（包括文件所在路径、文件名和后缀而不单单是文件所在目录）
    :param output_path: 最终输出目录
    :return 最终输出一个attr.csv文件
    """
    attr = pd.read_csv(r'' + attr_txt_path, sep='\t', header=None)
    attr.columns = ['linkID', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedLimit', 'level', 'width']
    attr['speedLimit'] = attr['speedLimit'].apply(lambda x: math.ceil(float(x) * 3.6))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    attr.to_csv(os.path.join(output_path, 'attr.csv'),
                index=False, mode='w', encoding='utf-8')


def concat_multi_history_feature(traffic_csv_dir: str,
                                 start_date: datetime.datetime = datetime.datetime(
                                     year=2019, month=6, day=1
                                 ),
                                 time_length: int = 60,
                                 output_dir: str = './concat_output_result'):
    """
    * 作用：合并同一天的多个feature的csv
    * 使用方法：
    ```
        files_list = get_file_list('./output_result')
        concat_multi_history_feature(files_list, './output_result')
    ```
    :param traffic_csv_dir: 各个时间点的recent数据所在的文件目录
    :param start_date: 程序开始遍历的日期，默认为2019年6月1日
    :param time_length: 程序遍历日期的长度，默认为60天，结合默认开始日期`start_date`，程序默认停止的时间为2019年7月31日
    :param output_dir: 最终结果保存的路径，默认为当前目录下的output_result文件夹
    :return 最终按照slice输出对应的合并文件
    """
    current_date = start_date
    files_list = get_file_list(traffic_csv_dir)
    for _ in range(time_length):
        # 生成当天的正则表达式
        current_date += datetime.timedelta(days=1)
        regex_str = '^{}{}{}_.*_feature.csv$'.format(
            current_date.year,
            format_with_zero(current_date.month),
            format_with_zero(current_date.day),
        )
        current_date_file_list = list()
        # 从文件列表中查找是否有符合的文件
        for file_name in files_list:
            match_file = re.match(regex_str, file_name)  # 匹配失败则返回None
            if match_file:
                current_date_file_list.append(file_name)

        if len(current_date_file_list) == 0:
            continue

        print(current_date_file_list)
        # 读取同一天的csv
        df_list = list()
        for file_name in current_date_file_list:
            path = os.path.join(traffic_csv_dir, file_name)
            df_list.append(pd.read_csv(path))

        # 将所有的csv合并在一起
        df_result = pd.concat(df_list, axis=0, ignore_index=True)
        del df_list
        # 去重
        df_result = df_result.drop_duplicates()
        # 指定`slice_id`列为数字列
        df_result['slice_id'] = df_result['slice_id'].apply(pd.to_numeric)
        # 重新排序
        df_result.sort_values(by='slice_id')  # 按照slice_id进行排序
        csv_name = r"{}{}{}_concat_feature.csv".format(
            current_date.year,
            format_with_zero(current_date.month),
            format_with_zero(current_date.day),
        )
        # 将结果保存为新的csv文件
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        df_result.to_csv(os.path.join(output_dir, csv_name), index=False)
        del df_result


def get_all_link_id(two_column_link_file_path: str) -> pd.Series:
    """
    * 作用：将获取的link一对一上（下）游关系表的两列拼合在一起之后去重，获得所有linkID的列表
    :param two_column_link_file_path: 具有两列linkid拓扑对应关系的文件
    :return: pd.Series
    """
    total_df = pd.read_csv(two_column_link_file_path)
    columns_name = total_df.columns.values
    first_column = total_df[columns_name[0]]
    second_column = total_df[columns_name[1]]
    return first_column.append(second_column)


def count_max_carrying_number(link_id_list: pd.Series, features_path: str, output_path: str = './max_carry_count'):
    """
    * 作用：统计每一天中每个link在三种状态下的最大承载量：即某种路况下参与统计的车辆数
    * 使用方法：
    ```
        result_list = get_all_link_id('../processed_data/topo_upstream.csv')  # 获取所有link的列表
        count_max_carrying_number(result_list, features_path)
    ```
    :param link_id_list: 所有需要统计的link列表（pd.Series）
    :param features_path: 所有concat_feature的文件所在目录
    :param output_path: 最后生成的结果文件目录（默认为当前目录）
    :return: 输出为四个文件，分别为label=0、1、2、3的时候的各个link的最大承载量
    """

    def count_row_feature(_row):
        if int(_row['link']) not in result_list[int(_row['label'])]:
            result_list[int(_row['label'])].setdefault(int(_row['link']), {
                'car_count': int(_row['car_count']),
                'count_times': 1,
            })
        else:
            # 如果新采集的通行车辆数量比记录值大，则替换掉它
            # 并更新统计次数
            if result_list[int(_row['label'])][int(_row['link'])]['car_count'] < int(_row['car_count']):
                result_list[int(_row['label'])][int(_row['link'])]['car_count'] = int(_row['car_count'])
                result_list[int(_row['label'])][int(_row['link'])]['count_times'] += 1

    assert type(link_id_list) == pd.Series, \
        print("The type of input link_id_list cannot be {} but require pd.Series.".format(type(link_id_list)))
    # 初始化结果列表（向结果中添加四个空缺列表）
    result_list = list()
    for _ in range(4):
        result_list.append(dict())
    # 遍历给定的feature文件
    files_list = get_file_list(features_path)
    bar = tqdm(total=len(files_list))
    for file_name in files_list:
        feature_df = pd.read_csv(os.path.join(features_path, file_name))
        for _, row in feature_df.iterrows():
            count_row_feature(row)
        bar.update()

    # 遍历结果列表中的字典，将其转换为list形式，使之满足pd.to_csv的格式
    max_carry_headers = ["link", "max_carry_num", "count_times"]
    for label_index, label_item in enumerate(result_list):
        result = list()
        for _, item in enumerate(label_item):
            result.append({
                "link": item,
                "max_carry_num": label_item[item]['car_count'],
                "count_times": label_item[item]['count_times'],
            })
        result.sort(key=take_link_id)  # 按照link进行排序
        max_carry_num = pd.DataFrame(result)
        # 生成结果文件
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        max_carry_num.to_csv(os.path.join(output_path, 'label_' + str(label_index) + '_max_carry.csv'),
                             header=max_carry_headers,
                             index=False, mode='w', encoding='utf-8')


def topo_draw(topo_one_to_one_file_path: str, attr_file_path: str):
    """
    * 作用：绘制路网拓扑图（运行太慢了，不建议使用）
    :param topo_one_to_one_file_path: 上（下）游的一对一拓扑关系文件路径（包括文件所在目录、文件名和后缀）
    :param attr_file_path: 道路固有属性文件路径（包括文件所在目录、文件名和后缀） ，必须是处理后的csv文件
    :return:
    """
    # 输入的topo结构需要是一对一的两列
    topo_df = pd.read_csv(topo_one_to_one_file_path, dtype=str)
    topo_cols_name = topo_df.columns.values
    # 获取路网属性
    attr_df = pd.read_csv(attr_file_path, dtype=str)
    total_link = attr_df['linkID']
    # 获取所有link的list
    total_link_list = total_link.tolist()
    # 获取link关系list，其中每个元素分别为[('A', 'B')]
    topo_first_column_list = topo_df[topo_cols_name[0]].tolist()
    topo_second_column_list = topo_df[topo_cols_name[1]].tolist()
    assert len(topo_first_column_list) == len(topo_second_column_list), \
        print("The topo relationship must be one to one")
    edges = list()
    for index in range(len(topo_first_column_list)):
        edges.append((topo_first_column_list[index], topo_second_column_list[index]))

    # =================================================================
    # 绘图
    graph = nx.Graph()
    for node in total_link_list:
        graph.add_node(node)
    r = graph.add_edges_from(edges)
    nx.draw(graph, with_labels=True, node_color='y', )
    plt.show()


def transform_attr_speed_limit_unit(attr_txt_path: str, output_path: str = './'):
    """
    * 作用：将attr.csv中link限速的单位换算：从m/s转换为km/h
    :param attr_txt_path: attr.csv文件所在的路径（包括文件所在路径、文件名和后缀而不单单是文件所在目录）
    :param output_path: 最终输出目录
    :return 最终输出一个attr.csv文件
    """
    attr = pd.read_csv(r'' + attr_txt_path, sep='\t', header=None)
    attr.columns = ['linkID', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedLimit', 'level',
                    'width']
    attr['speedLimit'] = attr['speedLimit'].apply(lambda x: math.ceil(float(x) * 3.6))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    attr.to_csv(os.path.join(output_path, 'attr.csv'),
                index=False, mode='w', encoding='utf-8')


def concat_multi_history_feature(traffic_csv_dir: str,
                                 start_date: datetime.datetime = datetime.datetime(
                                     year=2019, month=6, day=1
                                 ),
                                 time_length: int = 60,
                                 output_dir: str = './concat_output_result'):
    """
    * 作用：合并同一天的多个feature的csv
    * 使用方法：
    ```
        files_list = get_file_list('./output_result')
        concat_multi_history_feature(files_list, './output_result')
    ```
    :param traffic_csv_dir: 各个时间点的recent数据所在的文件目录
    :param start_date: 程序开始遍历的日期，默认为2019年6月1日
    :param time_length: 程序遍历日期的长度，默认为60天，结合默认开始日期`start_date`，程序默认停止的时间为2019年7月31日
    :param output_dir: 最终结果保存的路径，默认为当前目录下的output_result文件夹
    :return 最终按照slice输出对应的合并文件
    """
    current_date = start_date
    files_list = get_file_list(traffic_csv_dir)
    for _ in range(time_length):
        # 生成当天的正则表达式
        current_date += datetime.timedelta(days=1)
        regex_str = '^{}{}{}_.*_feature.csv$'.format(
            current_date.year,
            format_with_zero(current_date.month),
            format_with_zero(current_date.day),
        )
        current_date_file_list = list()
        # 从文件列表中查找是否有符合的文件
        for file_name in files_list:
            match_file = re.match(regex_str, file_name)  # 匹配失败则返回None
            if match_file:
                current_date_file_list.append(file_name)

        if len(current_date_file_list) == 0:
            continue

        print(current_date_file_list)
        # 读取同一天的csv
        df_list = list()
        for file_name in current_date_file_list:
            path = os.path.join(traffic_csv_dir, file_name)
            df_list.append(pd.read_csv(path))

        # 将所有的csv合并在一起
        df_result = pd.concat(df_list, axis=0, ignore_index=True)
        del df_list
        # 去重
        df_result = df_result.drop_duplicates()
        # 指定`slice_id`列为数字列
        df_result['slice_id'] = df_result['slice_id'].apply(pd.to_numeric)
        # 重新排序
        df_result.sort_values(by='slice_id')  # 按照slice_id进行排序
        csv_name = r"{}{}{}_concat_feature.csv".format(
            current_date.year,
            format_with_zero(current_date.month),
            format_with_zero(current_date.day),
        )
        # 将结果保存为新的csv文件
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        df_result.to_csv(os.path.join(output_dir, csv_name), index=False)
        del df_result


def get_all_link_id(two_column_link_file_path: str) -> pd.Series:
    """
    * 作用：将获取的link一对一上（下）游关系表的两列拼合在一起之后去重，获得所有linkID的列表
    :param two_column_link_file_path: 具有两列linkid拓扑对应关系的文件
    :return: pd.Series
    """
    total_df = pd.read_csv(two_column_link_file_path)
    columns_name = total_df.columns.values
    first_column = total_df[columns_name[0]]
    second_column = total_df[columns_name[1]]
    return first_column.append(second_column)


def count_max_carrying_number(link_id_list: pd.Series, features_path: str, output_path: str = './max_carry_count'):
    """
    * 作用：统计每一天中每个link在三种状态下的最大承载量：即某种路况下参与统计的车辆数
    * 使用方法：
    ```
        result_list = get_all_link_id('../processed_data/topo_upstream.csv')  # 获取所有link的列表
        count_max_carrying_number(result_list, features_path)
    ```
    :param link_id_list: 所有需要统计的link列表（pd.Series）
    :param features_path: 所有concat_feature的文件所在目录
    :param output_path: 最后生成的结果文件目录（默认为当前目录）
    :return: 输出为四个文件，分别为label=0、1、2、3的时候的各个link的最大承载量
    """

    def count_row_feature(_row):
        if int(_row['link']) not in result_list[int(_row['label'])]:
            result_list[int(_row['label'])].setdefault(int(_row['link']), {
                'car_count': int(_row['car_count']),
                'count_times': 1,
            })
        else:
            # 如果新采集的通行车辆数量比记录值大，则替换掉它
            # 并更新统计次数
            if result_list[int(_row['label'])][int(_row['link'])]['car_count'] < int(_row['car_count']):
                result_list[int(_row['label'])][int(_row['link'])]['car_count'] = int(_row['car_count'])
                result_list[int(_row['label'])][int(_row['link'])]['count_times'] += 1

    assert type(link_id_list) == pd.Series, \
        print("The type of input link_id_list cannot be {} but require pd.Series.".format(type(link_id_list)))
    # 初始化结果列表（向结果中添加四个空缺列表）
    result_list = list()
    for _ in range(4):
        result_list.append(dict())
    # 遍历给定的feature文件
    files_list = get_file_list(features_path)
    bar = tqdm(total=len(files_list))
    for file_name in files_list:
        feature_df = pd.read_csv(os.path.join(features_path, file_name))
        for _, row in feature_df.iterrows():
            count_row_feature(row)
        bar.update()

    # 遍历结果列表中的字典，将其转换为list形式，使之满足pd.to_csv的格式
    max_carry_headers = ["link", "max_carry_num", "count_times"]
    for label_index, label_item in enumerate(result_list):
        result = list()
        for _, item in enumerate(label_item):
            result.append({
                "link": item,
                "max_carry_num": label_item[item]['car_count'],
                "count_times": label_item[item]['count_times'],
            })
        result.sort(key=take_link_id)  # 按照link进行排序
        max_carry_num = pd.DataFrame(result)
        # 生成结果文件
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        max_carry_num.to_csv(os.path.join(output_path, 'label_' + str(label_index) + '_max_carry.csv'),
                             header=max_carry_headers,
                             index=False, mode='w', encoding='utf-8')


def topo_draw(topo_one_to_one_file_path: str, attr_file_path: str):
    """
    * 作用：绘制路网拓扑图（运行太慢了，不建议使用）
    :param topo_one_to_one_file_path: 上（下）游的一对一拓扑关系文件路径（包括文件所在目录、文件名和后缀）
    :param attr_file_path: 道路固有属性文件路径（包括文件所在目录、文件名和后缀） ，必须是处理后的csv文件
    :return:
    """
    # 输入的topo结构需要是一对一的两列
    topo_df = pd.read_csv(topo_one_to_one_file_path, dtype=str)
    topo_cols_name = topo_df.columns.values
    # 获取路网属性
    attr_df = pd.read_csv(attr_file_path, dtype=str)
    total_link = attr_df['linkID']
    # 获取所有link的list
    total_link_list = total_link.tolist()
    # 获取link关系list，其中每个元素分别为[('A', 'B')]
    topo_first_column_list = topo_df[topo_cols_name[0]].tolist()
    topo_second_column_list = topo_df[topo_cols_name[1]].tolist()
    assert len(topo_first_column_list) == len(topo_second_column_list), \
        print("The topo relationship must be one to one")
    edges = list()
    for index in range(len(topo_first_column_list)):
        edges.append((topo_first_column_list[index], topo_second_column_list[index]))

    # =================================================================
    # 绘图
    graph = nx.Graph()
    for node in total_link_list:
        graph.add_node(node)
    r = graph.add_edges_from(edges)
    nx.draw(graph, with_labels=True, node_color='y', )
    plt.show()


def get_slice_bucket_list(bucket_metric: int = 2, total_slice_length: int = 724) -> list:
    """
    * 作用：获取一个列表，每个元素为一个二维列表，代表了slice数据桶的开始和结束点
    :param bucket_metric: 每个数据桶的长度，默认为2，即每个数据桶中有2个slice
    :param total_slice_length: 总的时间长度，默认为724，即slice_id从0~723
    :return: 包含整个slice分桶的列表
    """

    def check_num_is_int(num) -> bool:
        assert type(num) in [int, float], \
            f"The type of given num for check must be 'int' or 'float' but not {type(num)}."
        return num - int(num) == 0

    assert type(bucket_metric) == int, f"The type of given bucket_metric must be int but not {type(bucket_metric)}."
    assert check_num_is_int(total_slice_length / bucket_metric), \
        "Given total_slice_length(default is 724) cannot be divided by bucket_metric."

    result = list()
    bucket_capacity = bucket_metric - 1  # 如果bucket_metric是2，则会每次获取的步长为3，如[0, 2]，即0、1、2，所以每个桶的容量（capacity）需要减掉1
    for start_slice in range(0, total_slice_length, bucket_metric):
        end_slice = start_slice + bucket_capacity
        result.append([start_slice, end_slice])

    return result


def operate_json(file_path: str, content=None) -> dict:
    """
    对json文件进行读写操作
    :param file_path: json文件地址
    :param content: 如果该值不为空则为“覆盖写”(w)模式，否则为“读”(r)模式
    :return: json文件的字典形式
    """
    operate_mode = 'r' if content is None else 'w'
    with codecs.open(file_path, operate_mode, 'utf-8') as f:
        if operate_mode == 'w':
            json.dump(content, f, ensure_ascii=False)
        else:
            return json.load(f)


def get_link_web_portrait(concat_result_dir: str, link_attr_path: str, link_topo_path: str,
                          output_dir: str = './link_web_portrait_result',
                          bucket_range: int = 2, total_slice_length: int = 724):
    """
    * 作用：获取指定时间片id下，在当前时间片下当前link及其上（下）游的属性及路况（如果在当前时间片id下存在的话），即
        “在某一指定时间片下，以某一link为中心的路网子系统的道路画像”
    * 程序流程：
        1. 从指定的concat文件夹中获取所有concat文件的文件名列表；
        2. 遍历所有concat文件，并执行如下操作：
            1. 遍历指定的时间片列表，锚定某个时间片slice_id，并从当前整个concat文件的DataFrame中根据该slice_id截取出一个小的DataFrame；
            2. 遍历这个特定时间片slice_id的DataFrame的每一行，假设某一行为row，则记录该行对应的link（记为“中心link”），并执行如下操作：
                1. 从topo文件中查询该link的上（下）游的link_id的list，并遍历该list，在上述“指定slice_id下的小的DataFrame”中执行另一次遍历；
                    1. 查看在该slice_id下是否存在当前中心link的上（下）游“真值”数据，如果有则压入结果列表中，没有则跳过
            3. 最终该slice_id输出一个csv，包含中心link的属性、中心link的上（下）游的link的属性、
                当前slice_id下的中心link的路况 以及 某个存在于当前slice_id中的link的路况
    * 操作须知：由于本方法运行缓慢，全部运行完毕大约需要109小时，因此内嵌了断点方法；如果需要中途停止运行，
                只需要在与本文件（data_utils.py）同目录下创建一个`stop.txt`即可停止，下次运行时本方法将会从断点的地方重新开始。
    :param concat_result_dir: concat文件所在目录
    :param link_attr_path: attr.csv所在路径（包含文件所在目录、文件名和文件后缀），必须是处理后的csv文件
    :param link_topo_path: topo.csv所在路径（包含文件所在目录、文件名和文件后缀），必须是处理后的csv文件
    :param output_dir: 输出路径，默认是在当前目录下的'link_web_portrait_result'文件夹下输出
    :param bucket_range: 每个数据桶的长度，默认为2，即每个数据桶中有2个slice
    :param total_slice_length: 总的时间长度，默认为724，即slice_id从0~723
    :return: 输出所需slice_id的csv文件，命名方式为“{slice_id_bucket}_train_data.csv”
    """
    # 读取相关的属性文件
    print('==> Start reading attr and topo files...')
    link_attr = pd.read_csv(link_attr_path)
    link_attr.drop(labels='level', axis=1, inplace=True)
    link_topo = pd.read_csv(link_topo_path)
    link_topo_column_name = link_topo.columns.values
    print('==> Reading attr and topo files is ready.')

    # 读取程序运行进度文件
    print('==> Start reading schedule files...')
    schedule_json_path = os.path.join(output_dir, 'schedule_{}_{}.json'.format(bucket_range, total_slice_length))
    if not os.path.exists(schedule_json_path):
        concat_result_dir = '../processed_data/concat_result'
        concat_result_list = get_file_list(concat_result_dir)
        schedule_point_json = {
            'bucket_range': bucket_range,
            'total_slice_length': total_slice_length,
            'last_run_point': '',
            'last_run_schedule': 0,
            'completed_slice': dict()
        }
        for day in concat_result_list:
            schedule_point_json['completed_slice'].setdefault(
                day, list()
            )
        operate_json(schedule_json_path, schedule_point_json)
    schedule_json = operate_json(schedule_json_path)
    print('==> Reading schedule files is ready.')
    if not schedule_json['last_run_point'] == '':
        print('==> Last run to' + schedule_json['last_run_point'])

    # 遍历所有数据文件
    concat_result_list = get_file_list(concat_result_dir)
    # 获取分桶的时间片列表
    slice_bucket = get_slice_bucket_list(bucket_range, total_slice_length)
    bar = tqdm(total=len(concat_result_list) * len(slice_bucket), initial=schedule_json['last_run_schedule'])
    for file_name in concat_result_list:
        concat_data = pd.read_csv(os.path.join(concat_result_dir, file_name))
        # 遍历指定的时间片
        for slice_id_bucket in slice_bucket:
            concat_data_point_slice_id = pd.DataFrame()
            # 检查该时间片是否已分析过
            if slice_id_bucket[0] in schedule_json['completed_slice'][file_name]:
                continue
            # 如果没有分析过则进入分析程序
            for slice_id in slice_id_bucket:
                concat_data_point_slice_id = pd.concat(
                    [concat_data[concat_data['slice_id'] == slice_id],
                     concat_data_point_slice_id],
                    axis=0, ignore_index=True
                )
            if len(concat_data_point_slice_id) == 0:
                continue
            # 初始化结果列表
            result_list = list()
            # 遍历从concat文件中根据当前指定的slice_id截取的数据块
            for _, concat_row in concat_data_point_slice_id.iterrows():
                # 当前link的ID
                current_link = concat_row['link']
                # 遍历当前link的上（下）游信息
                current_link_topo = link_topo[link_topo[link_topo_column_name[0]] == current_link]
                # 当前link的上（下）游信息dict
                link_info_dict = {
                    # 当前中心link的路况
                    'current_link_attr_link': 0,
                    'current_link_status_speed': 0,
                    'current_link_status_eta_speed': 0,
                    'current_link_status_label': 0,
                    'current_link_status_car_count': 0,
                    # 当前中心link的自身道路属性
                    'current_link_attr_length': 0,
                    'current_link_attr_direction': 0,
                    'current_link_attr_pathclass': 0,
                    'current_link_attr_speedclass': 0,
                    'current_link_attr_LaneNum': 0,
                    'current_link_attr_speedLimit': 0,
                    'current_link_attr_width': 0,
                    # 当前中心link的上（下）游拓扑道路属性
                    # 第1个上（下）游道路属性
                    'topo_link_1_link': -1,
                    'topo_link_1_length': 0,
                    'topo_link_1_direction': 0,
                    'topo_link_1_pathclass': 0,
                    'topo_link_1_speedclass': 0,
                    'topo_link_1_LaneNum': 0,
                    'topo_link_1_speedLimit': 0,
                    'topo_link_1_width': 0,
                    # 第2个上（下）游道路属性
                    'topo_link_2_link': 0,
                    'topo_link_2_length': 0,
                    'topo_link_2_direction': 0,
                    'topo_link_2_pathclass': 0,
                    'topo_link_2_speedclass': 0,
                    'topo_link_2_LaneNum': 0,
                    'topo_link_2_speedLimit': 0,
                    'topo_link_2_width': 0,
                    # 第3个上（下）游道路属性
                    'topo_link_3_link': 0,
                    'topo_link_3_length': 0,
                    'topo_link_3_direction': 0,
                    'topo_link_3_pathclass': 0,
                    'topo_link_3_speedclass': 0,
                    'topo_link_3_LaneNum': 0,
                    'topo_link_3_speedLimit': 0,
                    'topo_link_3_width': 0,
                    # 第4个上（下）游道路属性
                    'topo_link_4_link': 0,
                    'topo_link_4_length': 0,
                    'topo_link_4_direction': 0,
                    'topo_link_4_pathclass': 0,
                    'topo_link_4_speedclass': 0,
                    'topo_link_4_LaneNum': 0,
                    'topo_link_4_speedLimit': 0,
                    'topo_link_4_width': 0,
                    # 第5个上（下）游道路属性
                    'topo_link_5_link': 0,
                    'topo_link_5_length': 0,
                    'topo_link_5_direction': 0,
                    'topo_link_5_pathclass': 0,
                    'topo_link_5_speedclass': 0,
                    'topo_link_5_LaneNum': 0,
                    'topo_link_5_speedLimit': 0,
                    'topo_link_5_width': 0,
                    # 第6个上（下）游道路属性
                    'topo_link_6_link': 0,
                    'topo_link_6_length': 0,
                    'topo_link_6_direction': 0,
                    'topo_link_6_pathclass': 0,
                    'topo_link_6_speedclass': 0,
                    'topo_link_6_LaneNum': 0,
                    'topo_link_6_speedLimit': 0,
                    'topo_link_6_width': 0,
                }
                for _, link_relationship in current_link_topo.iterrows():
                    # 这是单条上（下）游的信息
                    truth_link_df = concat_data_point_slice_id[
                        concat_data_point_slice_id['link'] == link_relationship[link_topo_column_name[1]]]
                    # 如果当前link在当前时间片下存在其上（下）游的link真值
                    if not len(truth_link_df) == 0:
                        single_truth_link_dict = dict()
                        # 待预测的上游linkID 以及 待预测的上游link路况（真值）
                        single_truth_link_dict.setdefault('stream_link_id', truth_link_df.iloc[0].at['link'])
                        single_truth_link_dict.setdefault('speed', truth_link_df.iloc[0].at['speed_with_light'])
                        single_truth_link_dict.setdefault('eta_speed', truth_link_df.iloc[0].at['speed_without_light'])
                        single_truth_link_dict.setdefault('zero_label',
                                                          1 if truth_link_df.iloc[0].at['label'] == 0 else 0)
                        single_truth_link_dict.setdefault('unobstructed_label',
                                                          1 if truth_link_df.iloc[0].at['label'] == 1 else 0)
                        single_truth_link_dict.setdefault('slow_label',
                                                          1 if truth_link_df.iloc[0].at['label'] == 2 else 0)
                        single_truth_link_dict.setdefault('congested_label',
                                                          1 if truth_link_df.iloc[0].at['label'] == 3 else 0)
                        single_truth_link_dict.setdefault('car_count', truth_link_df.iloc[0].at['car_count'])

                        # 目标link的上（下）游link（去除倒数第二个特征：level）
                        # 如果在此之前还没有初始化当前link及其上（下）游属性字典
                        if link_info_dict['topo_link_1_link'] == -1:
                            for row_index in range(len(current_link_topo)):
                                # 当前上（下）游的linkID
                                stream_link_id = current_link_topo.iloc[row_index].at[link_topo_column_name[1]]
                                # 当前上（下）游的属性
                                stream_link_attr = link_attr[link_attr['linkID'] == stream_link_id]
                                # 初始化上（下）游属性字典
                                link_info_dict[f'topo_link_{row_index + 1}_link'] = stream_link_id
                                link_info_dict[f'topo_link_{row_index + 1}_length'] = stream_link_attr.iloc[0].at[
                                    'length']
                                link_info_dict[f'topo_link_{row_index + 1}_direction'] = stream_link_attr.iloc[0].at[
                                    'direction']
                                link_info_dict[f'topo_link_{row_index + 1}_pathclass'] = stream_link_attr.iloc[0].at[
                                    'pathclass']
                                link_info_dict[f'topo_link_{row_index + 1}_speedclass'] = stream_link_attr.iloc[0].at[
                                    'speedclass']
                                link_info_dict[f'topo_link_{row_index + 1}_LaneNum'] = stream_link_attr.iloc[0].at[
                                    'LaneNum']
                                link_info_dict[f'topo_link_{row_index + 1}_speedLimit'] = stream_link_attr.iloc[0].at[
                                    'speedLimit']
                                link_info_dict[f'topo_link_{row_index + 1}_width'] = stream_link_attr.iloc[0].at[
                                    'width']
                            # 当前link的属性
                            current_link_attr = link_attr[link_attr['linkID'] == current_link]
                            link_info_dict['current_link_attr_link'] = current_link
                            link_info_dict['current_link_attr_length'] = current_link_attr.iloc[0].at['length']
                            link_info_dict['current_link_attr_direction'] = current_link_attr.iloc[0].at['direction']
                            link_info_dict['current_link_attr_pathclass'] = current_link_attr.iloc[0].at['pathclass']
                            link_info_dict['current_link_attr_speedclass'] = current_link_attr.iloc[0].at['speedclass']
                            link_info_dict['current_link_attr_LaneNum'] = current_link_attr.iloc[0].at['LaneNum']
                            link_info_dict['current_link_attr_speedLimit'] = current_link_attr.iloc[0].at['speedLimit']
                            link_info_dict['current_link_attr_width'] = current_link_attr.iloc[0].at['width']
                            # 当前link的路况（真值）
                            current_link_truth = concat_data_point_slice_id[
                                concat_data_point_slice_id['link'] == current_link]
                            link_info_dict['current_link_status_speed'] = current_link_truth.iloc[0].at[
                                'speed_with_light']
                            link_info_dict['current_link_status_eta_speed'] = current_link_truth.iloc[0].at[
                                'speed_without_light']
                            link_info_dict['current_link_status_label'] = current_link_truth.iloc[0].at['label']
                            link_info_dict['current_link_status_car_count'] = current_link_truth.iloc[0].at['car_count']
                        # 将当前link及上（下）游属性字典压入结果列表中
                        for _, item_name in enumerate(link_info_dict):
                            single_truth_link_dict.setdefault(item_name, link_info_dict[item_name])

                        # 将与当前节点有关的所有信息全部压入结果list中
                        result_list.append(single_truth_link_dict)
                    else:
                        continue

            # 输出
            pointed_slice_id_df = pd.DataFrame(result_list)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_file_name = "{}to{}(Slice Bucket {})_web_portrait.csv".format(slice_id_bucket[0],
                                                                                 slice_id_bucket[1],
                                                                                 bucket_range)
            pointed_slice_id_df.to_csv(os.path.join(output_dir, output_file_name),
                                       index=False, mode='a', encoding='utf-8')
            # 更新进度文件
            for slice_id in slice_id_bucket:
                schedule_json['completed_slice'][file_name].append(slice_id)
            schedule_json['last_run_schedule'] += 1
            bar.update()
            # 检查是否停止运行程序
            if os.path.exists('./stop.txt'):
                print('==> User stop the program, then the breakpoint info will be recorded.')
                schedule_json['last_run_point'] = 'File Name: {}, Slice Bucket: from {} to {}'.format(
                    file_name, slice_id_bucket[0], slice_id_bucket[1]
                )
                operate_json(schedule_json_path, schedule_json)
                print('==> Breakpoint info is recorded, the program will be stopped automatically.')
                return
