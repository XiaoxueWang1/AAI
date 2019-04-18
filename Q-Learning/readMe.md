## 该文件包含三部分，分别是：
1 testRLAgent.py文件的改动
2 文件夹中各个文件的用意
3 如何运行代码(重要)

1 testRLAgent.py文件的改动：
testRLAgent.py仅用于对模型进行测试，改动的代码均有注释。
分别为：
14-18行，加载训练的模型；
20-22行，定义总的Score；
35-37行，计算总的Score；
43-45行，输出Score等信息

2 文件夹中各个文件的用意：
Agent.py 和 testRLAgent.py为作业要求文件，不再赘述
readme.txt 此文件，用于说明
model.json 训练好的Q-table，运行时必须在文件夹内
trainRLAgent.py用于训练模型，调用trainAgent类中的方法

3 如何运行代码(重要)
相关库：json，gym，datetime，numpy，random
运行测试文件时，必须将model.json，Agent.py，testRLAgent.py放入同一目录下(重要)
运行命令：python testRLAgent.py
运行训练文件时，必须将model.json，Agent.py，trainRLAgent.py放入同一目录下(重要)
运行命令：python trainRLAgent.py
