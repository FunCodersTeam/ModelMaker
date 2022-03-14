### 安装
```
> pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```
### 帮助
```
> python trainer.py -h
```
### 开始使用
```
> python trainer.py --input "D:\\dataset\\test"
```
生成的tfjs_model/model.json、tfjs_model/group1-shard1of1.bin、labels.txt文件可以在智能车模拟器中导入测试
### 数据集目录结构
```
> tree
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