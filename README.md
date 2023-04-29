<a href="https://app.codacy.com/gh/FunCodersTeam/ModelMaker/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/77c9118814a34d1cb958940276f77caa"/></a>
### 安装
```bash
$ pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```
### 帮助
```bash
$ python trainer.py -h
```
### 开始使用
```bash
$ python trainer.py --input "D:\\dataset\\test"
```
生成的 `tfjs_model/model.json`、`tfjs_model/group1-shard1of1.bin`、`labels.txt` 文件可以在<a href="https://github.com/FunCodersTeam/WebCarSim">智能车模拟器</a>中导入测试
### 数据集目录结构
```bash
$ tree
└─test
    ├─榴莲
    ├─橙子
    ├─牛
    ├─狗
    ├─猪
    ├─猫
    ├─苹果
    ├─葡萄
    ├─香蕉
    └─马
```
