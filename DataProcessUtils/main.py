from topo import *
from data_utils import *
import time

# 各个原始文件所在的路径（或目录）
traffic_txt_dir = '../orginal_data/traffic-fix/traffic'
attr_txt_path = '../orginal_data/attr.txt'
topo_txt_path = '../orginal_data/topo.txt'

# 各个输出文件所在路径（或目录）
all_result = './result'
traffic_csv_dir_ = 'traffic_to_csv_output'
concat_result_dir_ = 'concat_output_result'
# 对输出路径进行合并操作等
if not os.path.exists(all_result):
    os.mkdir(all_result)
traffic_csv_dir = os.path.join(all_result, traffic_csv_dir_)
concat_result_dir = os.path.join(all_result, concat_result_dir_)
attr_csv_path = os.path.join(all_result, 'attr.csv')
topo_downstream_one2one_csv_path = os.path.join(all_result, 'topo_downstream_ono2one.csv')
topo_upstream_one2one_csv_path = os.path.join(all_result, 'topo_upstream_ono2one.csv')
topo_join_attr_path = os.path.join(all_result, 'topo_join_attr.csv')

time_start_all = time.time()  # 程序运行总计时

# 获取所有所需的格式化文件
#  =====================================================
# 1. 将所有traffic文件转为csv并进行分块（非常慢）
print("-[ ] 1. Start: 将所有traffic文件转为csv并进行分块 ...")
time_start = time.time()
file_list = get_file_list(traffic_txt_dir)
bar = tqdm(total=len(file_list))
for item in file_list:
    transform_file_to_csv(traffic_txt_dir, item, traffic_csv_dir)
    bar.update()
time_end = time.time()
print("-[x] 1. End: 完成！耗时：{}".format(time_end - time_start))

# 2. 将同属于一天的recent_feature和history_feature合并为一个文件并去重（非常慢）
print("-[ ] 2. Start: 将同属于一天的recent_feature和history_feature合并为一个文件并去重 ...")
time_start = time.time()
concat_multi_history_feature(traffic_csv_dir, output_dir=concat_result_dir)
time_end = time.time()
print("-[x] 2. End: 完成！耗时：{}".format(time_end - time_start))

# 3. 获取attr.csv并进行速度转换
print("-[ ] 3. Start: 获取attr.csv并进行速度转换 ...")
time_start = time.time()
transform_attr_speed_limit_unit(attr_txt_path, output_path=all_result)
time_end = time.time()
print("-[x] 3. End: 完成！耗时：{}".format(time_end - time_start))

# 4. 获取一对一的下游拓扑结构：topo_ono2one.csv
print("-[ ] 4. Start: 获取一对一的下游拓扑结构 ...")
time_start = time.time()
transform_one2many_to_one2one(topo_txt_path, output_dir=all_result)
time_end = time.time()
print("-[x] 4. End: 完成！耗时：{}".format(time_end - time_start))

# 5. 获取一对一的上游拓扑结构：topo_upstream.csv
print("-[ ] 5. Start: 获取一对一的上游拓扑结构 ...")
time_start = time.time()
get_upstream_one2one(topo_downstream_one2one_csv_path, output_dir=all_result)
time_end = time.time()
print("-[x] 5. End: 完成！耗时：{}".format(time_end - time_start))

# 6. 获取道路画像（非常慢）
print("-[ ] 6. Start: 获取道路画像 ...")
time_start = time.time()
time_list = [261]  # 所有的slice_id范围：[i for i in range(724)]
get_link_web_portrait(concat_result_dir, attr_csv_path, topo_upstream_one2one_csv_path,
                      time_list, output_dir=os.path.join(all_result, 'link_web_portrait_result'))
time_end = time.time()
print("-[x] 6. Start: 完成！耗时：{}".format(time_end - time_start))
# =====================================================

# 计算上下游link的direction的相关性（使用肯达尔kendall相关性分析）
#  =====================================================
# print("-[ ] 7. Start: 计算上下游link的direction的相关性 ...")
# time_start = time.time()
# # 将道路拓扑和道路属性连接在一起
# join_topo_and_attr(topo_downstream_one2one_csv_path, attr_csv_path, output_path=all_result)
# # 读取数据并进行分析
# df = pd.read_csv(r'' + topo_join_attr_path)
# columns_list = ['p_direction', 'c_direction']
# print(correlation_analysis(df, columns_list, 'kendall'))
# time_end = time.time()
# print("-[x] 7. End: 完成！耗时：{}".format(time_end - time_start))
#  =====================================================

# 统计某一条link的最大承载量（非常慢）
# =====================================================
# print("-[ ] 8. Start: 统计某一条link的最大承载量 ...")
# time_start = time.time()
# all_link_id_list = get_all_link_id(topo_downstream_one2one_csv_path)  # 获取所有link的列表
# count_max_carrying_number(all_link_id_list, concat_result_dir, output_path=os.path.join(all_result, 'max_carry_count'))
# time_end = time.time()
# print("-[x] 8. Start: 完成！耗时：{}".format(time_end - time_start))
# =====================================================

time_end_all = time.time()  # 程序运行总计时结束
print("# ==> 程序运行总计时：{}".format(time_end_all - time_start_all))

# 绘制路网拓扑图（极其慢，弃用）
# =====================================================
# topo_draw(topo_upstream_one2one_csv_path, attr_txt_path)
# =====================================================
