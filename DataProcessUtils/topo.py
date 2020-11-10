import pandas as pd
import os


def transform_one2many_to_one2one(file_path: str, output_dir: str = "./"):
    """
    * 作用：将拓扑结构从一对多转变为一对一关系
    :param file_path: topo.txt所在路径（必须是txt文件）
    :param output_dir: 结果文件的输出路径
    :return 最终输出“topo_ono2one.csv”文件
    """
    topo_list = list()
    topo_headers = ['currentLink', 'downstreamLink']
    with open(r'' + file_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_line_info = line.split('\t')
            current_link = all_line_info[0]
            downstream_link_list = all_line_info[1].split(',')
            for downstream_link in downstream_link_list:
                topo_list.append({
                    'currentLink': current_link,
                    'downstreamLink': downstream_link.rstrip("\n")
                })

        # 输出文件
        _data = pd.DataFrame(topo_list)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        _data.to_csv(os.path.join(output_dir, 'topo_downstream_ono2one.csv'),
                     header=topo_headers,
                     index=False, mode='w', encoding='utf-8')


def join_topo_and_attr(topo_csv_path: str, attr_csv_path: str, output_path: str = './'):
    """
    * 作用：将topo.csv和attr.csv连接在一起
    :param topo_csv_path: 拓扑文件所在路径（必须是csv）
    :param attr_csv_path: 道路属性文件所在路径（必须是csv）
    :param output_path: 最终文件输出路径
    :return 最终输出“topo_join_attr.csv”文件
    """
    topo = pd.read_csv(r'' + topo_csv_path)
    topo_columns = topo.columns.values
    attr = pd.read_csv(r'' + attr_csv_path)
    join_result = topo.join(attr.set_index('linkID'), on=topo_columns[0])
    join_result = join_result.rename(
        columns={'length': 'p_length',
                 'direction': "p_direction",
                 "pathclass": "p_pathclass",
                 "speedclass": "p_speedclass",
                 "LaneNum": "p_LaneNum",
                 "speedLimit": "p_speedLimit",
                 "level": "p_level",
                 "width": "p_width"})
    join_result = join_result.join(attr.set_index('linkID'), on=topo_columns[1])
    join_result = join_result.rename(
        columns={'length': 'c_length',
                 'direction': "c_direction",
                 "pathclass": "c_pathclass",
                 "speedclass": "c_speedclass",
                 "LaneNum": "c_LaneNum",
                 "speedLimit": "c_speedLimit",
                 "level": "c_level",
                 "width": "c_width"})
    _data = pd.DataFrame(join_result)
    _data.to_csv(os.path.join(output_path, 'topo_join_attr.csv'),
                 index=False, mode='w', encoding='utf-8')


def correlation_analysis(data_frame: pd.DataFrame, columns_list: list, mode: str = 'pearson'):
    """
    * 作用：判断给定的列在给定的DataFrame中的相关性
    :param data_frame: 指定的DataFrame
    :param columns_list: 指定的列名
    :param mode: 相关性判断模式，可选的有'pearson', 'kendall', 'spearman'三种，默认为'pearson'
    :return: 各个变量间的相关性矩阵
    """
    assert mode in ['pearson', 'kendall', 'spearman'], \
        "The selected mode must be 'kendall' or 'spearman', otherwise should not point mpde."

    return data_frame[columns_list].corr(mode)


def get_upstream_one2one(file_path: str, output_dir: str = "./"):
    """
    * 作用：从一对一的下游拓扑结构文件中获取上游的一对一拓扑关系
    :param file_path: 一对一下游拓扑结构文件所在路径
    :param output_dir: 最终文件输出路径
    :return: 输出“topo_upstream.csv”文件
    """
    topo_list = list()
    topo_headers = ['currentLink', 'upstreamLink']
    result_dict = dict()
    with open(r'' + file_path, 'r', encoding='utf-8') as f:
        # for line in f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            all_line_info = line.split(',')
            upstream_link = all_line_info[0]
            current_link = all_line_info[1]

            # 去除\n，否则输出的文件中含有`"`
            current_link = current_link.rstrip("\n")

            if current_link not in result_dict:
                result_dict.setdefault(current_link, [upstream_link])
            else:
                result_dict[current_link].append(upstream_link)

        for _, current_link in enumerate(result_dict):
            for _, upstream_link in enumerate(result_dict[current_link]):
                topo_list.append({
                    'currentLink': current_link,
                    'upstreamLink': upstream_link,
                })

        _data = pd.DataFrame(topo_list)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        _data.to_csv(os.path.join(output_dir, 'topo_upstream_ono2one.csv'),
                     header=topo_headers,
                     index=False, mode='w', encoding='utf-8')
