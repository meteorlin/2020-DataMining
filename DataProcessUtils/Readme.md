```
Version: 2020-11-10
Author: Meteorlin
```

# 使用方法
1. 打开main.py，将`traffic_txt_dir`、`attr_txt_path`和`topo_txt_path`分别修改为30个traffic.txt所在的目录、attr.txt文件所在的路径以及topo.txt文件所在的路径；
2. 运行main.py；

# 注意
1. 运行所有子方法的时间会比较长，大概在6到8个小时之间，如果只需要部分输出，请自行在main.py中注释不需要的子方法；
2. main.py中包含数据整理子方法（第1-6步）和分析子方法（第7-8步）。

# Update By Meteorlin in 2020-11-11 (for datautils.py)
1. I found that this method requires at least 109 HOURS to finish its functions, so I added a function that allows this method to stop halfway, which is called a **breakpoint record**. **NOTE** that if you want to stop this method halfway, **DO NOT** stop it via normal stopping way, but create a `txt` file names `stop` in the same path with this `.py` file.
2. After last meeting, I add a function that allows you to point a `slice range`. It will automatically split the whole time slice sequence by the given `slice range` (which means a time interval). **NOTE** that the given `silce range` parameter must be a common factor of 724 (the total length of all slice), otherwise the method will raise an exception (defined by myself).

> Best wishes for double 11 Festival!
