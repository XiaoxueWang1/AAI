# -*- coding: utf-8 -*-
import numpy as np
import copy

def randonmPro(pro): #以概率pro生成1，(1-pro)生成0
    pro *= 100
    num = np.random.randint(1,101)
    if num <= pro:
        return 1
    else:
        return 0

def sigmoid(z): #激活函数
    return 1/(1+np.exp(-z))

def createTrainingData():    # 返回训练数据集
    data = []
    for i in range(32):
        L = list(str(bin(i))[2:].zfill(5))
        L = [ float(i) for i in L ]
        data.append(L)
    labels = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,1,0,0,1,1,0,0,1,0,1,1,0]
    return data,labels

def init(N):  #隐藏结点数N初始化
    connectedMatrix = np.zeros([N + 1, m + N], int) #连接矩阵
    #connectedMatrix = np.random.randint(0, 2, (N + 1, m + N))
    derMatrix = np.zeros([N + 1, m + N + 1]) #偏导weight权值矩阵，初始化为0
    weightMatrix = np.random.uniform(-5, 5, (N + 1, m + N + 1)) #初始化权值矩阵，利用(-5,5)的均匀分布
    for i in range(N + 1):
        for j in range(m + N):
            connectedMatrix[i][j] = randonmPro(0.5) #以0.5的概率生成连接矩阵
            if connectedMatrix[i][j]==0:
                weightMatrix[i][j+1]=0
            if j >= i + m:
                connectedMatrix[i][j] = 0
                weightMatrix[i][j + 1] = 0
    return connectedMatrix,weightMatrix,derMatrix

class Network: #神经网络类
    def __init__(self, connectedMatrix, weightMatrix, derMatrix, m,N):
        self.connectedMatrix = connectedMatrix
        self.weightMatrix = weightMatrix
        self.derMatrix = derMatrix #偏导weight矩阵
        self.success = False  #标记网络是否成功
        self.m = m  #网络的输入结点个数
        self.N = N  #网络的隐藏结点个数
        self.a = [] #网络中每个结点的输出 a=g(z)
        self.z = [] #网络中每个结点的输入
        self.delta = [0]*(N+1) #L对每个结点(从隐藏结点到输出结点)输入的偏导 例如delta[N]表示L对z[m+N]的偏导
        self.E = 100 #E即算出的cost值，用作排序

    def clearderMatrix(self): #用于清空偏导矩阵，梯度下降时使用
        N = self.N
        self.derMatrix = np.zeros([N + 1, m + N + 1])

    def Pretreatment(self,inData): #预处理，讲delta清空，a，z均初始化为输出数据
        N = self.N
        self.a = []
        self.z = []
        self.delta = [0] * (N + 1)
        self.a.extend(inData)
        self.z.extend(inData)

    def forward(self): #前向传播，填充a和z
        N = self.N
        for i in range(N+1):
            s = -self.weightMatrix[i][0]
            for j in range(m+i):
                if self.connectedMatrix[i][j] == 1:
                    s += self.weightMatrix[i][j+1]*self.a[j]
            self.z.append(s)
            s = sigmoid(s)
            self.a.append(s)

    def derivatives(self,label): #计算偏导，即BP(后向传播)算法，填充偏导weight矩阵和delta
        N = self.N
        out = self.a[m + N]
        self.delta[N] = (out - label) * out * (1-out)
        self.derMatrix[N][0] += self.delta[N]*(-1) #填充bias偏导
        for j in range(m+N):
            if self.connectedMatrix[N][j] == 1:
                self.derMatrix[N][j+1] += self.delta[N]*self.a[j]
        i = N-1  # i表示第i个隐藏节点
        while i >= 0: #从最后一个结点往前
            for j in range(i+1,N+1):
                if self.connectedMatrix[j][m+i] == 1:
                    self.delta[i] += self.delta[j]*self.weightMatrix[j][m+i+1]*self.a[m+i]*(1-self.a[m+i])
            self.derMatrix[i][0] += self.delta[i]*(-1) #填充bias偏导
            for j in range(m + i):
                if self.connectedMatrix[i][j] == 1:
                    self.derMatrix[i][j + 1] += self.delta[i] * self.a[j]
            i -= 1

    def GradientDescent(self,rate): #梯度下降算法
        N = self.N
        temp = np.zeros([N + 1, m + N + 1])
        i = N  # i表示第i个隐藏节点
        while i >= 0:
            temp[i][0] = self.weightMatrix[i][0] - self.derMatrix[i][0]*rate
            for j in range(m + i):
                if self.connectedMatrix[i][j] == 1:
                    temp[i][j + 1] = self.weightMatrix[i][j + 1] - self.derMatrix[i][j + 1]*rate
            i -= 1
        self.weightMatrix = temp #每次同步更新
def partTrain(net,learningRate,iterNum): #部分训练方法，对神经网络部分训练
    N = net.N
    for times in range(iterNum):
        delta = 0 #记录cost
        delta1 = 0 #记录错误预测的个数
        #开始梯度下降过程
        net.clearderMatrix()
        for i in range(32):
            net.Pretreatment(trainingData[i])
            net.forward()
            net.derivatives(labels[i])
            if net.a[m + N] >= 0.5: #输出结果大于等于0.5则预测为1
                pre = 1
            else:  #小于0.5则预测为0
                pre = 0
            delta += 0.5*(net.a[m + N] - labels[i])*(net.a[m + N] - labels[i])
            delta1 += abs(pre - labels[i])
        if delta1 == 0: #预测正确跳出循环
            break
        if times == 0:
            initD = net.a[m+N] #首次记录初始的误差cost
        net.GradientDescent(learningRate)
    net.E = delta #网络的E表示该网络的cost
    return delta,delta1,initD

def SA(net): #模拟退火算法的变型算法
    tempNet = copy.deepcopy(net)#对原网络copy，保证结果较差时返回原网络
    initT = 1000 #初始温度
    minT = 1 #最小温度
    iterL = 20 #每个温度的迭代次数
    nowT = initT #当前温度
    initW = tempNet.weightMatrix #初始权值
    N = tempNet.N
    oldDelta = net.E
    while nowT > minT:
        for j in range(iterL): #每个温度迭代一定次数
            #newW = initW + np.random.randn(N + 1, m + N + 1) #突变1：用标准高斯分布
            #newW = initW + np.random.uniform(-5, 5, (N + 1, m + N + 1)) #突变2：用(-5,5)的均匀分布
            newW = np.random.uniform(-5, 5, (tempNet.N + 1, m + tempNet.N + 1)) #突变3：替换为(-5,5)的均匀分布；突变较大，成功概率大
            tempNet.weightMatrix = newW
            for i1 in range(tempNet.N + 1):
                for j1 in range(m + tempNet.N):
                    if tempNet.connectedMatrix[i1][j1] == 0:
                        tempNet.weightMatrix[i1][j1 + 1] = 0
                    if j1 >= i1 + m:
                        tempNet.connectedMatrix[i1][j1] = 0
                        tempNet.weightMatrix[i1][j1 + 1] = 0
            Delta, Delta1,initD = partTrain(tempNet, 0.5, 200) #对突变后进行测试
            if Delta1==0:
                return tempNet,Delta1
            res = Delta - oldDelta
            if res<0:  #扰动后效果好，接受这个解
                Delta, Delta1, initD = partTrain(tempNet, 0.5, 2000) #对新解部分训练
                initW = tempNet.weightMatrix
                oldDelta = Delta
                if Delta1 == 0: #没有出错，跳出循环
                    return tempNet, Delta1
            else: #与一般的SA不同，当解不好时不接受。只接受效果好的。这样一来只能依赖足够大的突变产生较好的解
                tempNet.weightMatrix = initW
        nowT -= 100
    tempNet.E = Delta
    if net.E - Delta > 0.5*net.E: #当E减少明显时，则接受这个
        tempNet.success = True
        return tempNet,Delta1
    else:
        net.success = False
        return net, net.E

def ifFull(net):
    full = True
    for i in range(net.N + 1):
        for j in range(net.m + net.N):
            if j < i + m:
                if net.connectedMatrix[i][j]==0: #存在未连接的边
                    full = False
    return full

def addConnections(net): #添加连接
    #net.connectedMatrix = np.random.randint(1, 2, (net.N + 1, net.m + net.N))
    for i in range(net.N + 1):
        for j in range(net.m + net.N):
            if net.connectedMatrix[i][j]==0: #对于未连接的边随机连接
                rand = randonmPro(0.7)
                if rand == 1:
                    net.connectedMatrix[i][j] = 1
                    net.weightMatrix[i][j + 1] = np.random.uniform(-5, 5) #新连接的边进行随机初始化
            if j >= i + m:
                net.connectedMatrix[i][j] = 0
                net.weightMatrix[i][j + 1] = 0

def reduceNodenum(net): #减少结点数
    net.N -= 1
    net.connectedMatrix = np.random.randint(0, 2, (net.N + 1, net.m + net.N))
    net.weightMatrix = np.random.uniform(-5, 5, (net.N + 1, net.m + net.N + 1))  # 初始化权值矩阵，利用(-5,5)的均匀分布
    for i in range(net.N + 1):
        for j in range(net.m + net.N):
            if net.connectedMatrix[i][j]==0:
                net.weightMatrix[i][j+1]=0
            if j >= i + m:
                net.connectedMatrix[i][j] = 0
                net.weightMatrix[i][j + 1] = 0

if __name__ == "__main__":
    m = 5 #输入node个数
    M = 20 #初始群体个数
    ifSucceed = False
    trainingData, labels = createTrainingData() #获取数据集
    NList = []

    for i in range(M):
        nodeNum = np.random.randint(2,5) #初始的结点个数为[2,5)
        NList.append(nodeNum)
    netList = []
    for N in NList:
        connectedMatrix, weightMatrix, derMatrix = init(N) #随机初始化
        net = Network(connectedMatrix,weightMatrix,derMatrix, m,N) #构建结点
        learningRate = 0.5 #学习率设定，没有演化
        delta, delta1, initD = partTrain(net,learningRate,2000) #进行部分演化
        if initD - delta > 0.3 * initD: #明显下降则成功，否则失败
            net.success = True
        else:
            net.success = False
        if delta1 == 0: #如果成功，则输出矩阵
            #print(net.weightMatrix)
            for row in range(net.N + 1):
                for col in range(m + net.N + 1):
                    print(str(net.weightMatrix[row][col])+' ', end='')
                print("\n", end='')
            ifSucceed = True
            break
        netList.append(net)

    generation = 1 #演化代数
    while ifSucceed == False: #成功则跳出循环
        netList.sort(key=lambda aNet: aNet.E) #根据E来排序
        net = netList[0]
        if net.success == True: #如果网络被标记为成功
            delta, delta1, initD = partTrain(net, learningRate, 100)
            if delta1 == 0: #可接受，跳出
                #print(net.weightMatrix)
                for row in range(net.N + 1):
                    for col in range(m + net.N + 1):
                        print(str(net.weightMatrix[row][col]) + ' ', end='')
                    print("\n", end='')
                ifSucceed = True
                break
            if initD - delta > 0.3 * initD:
                net.success = True
            else:
                net.success = False
        else:
            net,delta1 = SA(net) #BP达到局部最优后，进行SA算法
            if delta1 == 0: #可接受则输出
                #print(net.weightMatrix)
                for row in range(net.N + 1):
                    for col in range(m + net.N + 1):
                        print(str(net.weightMatrix[row][col]) + ' ', end='')
                    print("\n", end='')
                ifSucceed = True
                break
            if net.success == False: #BP与SA均失败，则更改结构
                if net.N == 2: #有两个node
                   if ifFull(net) == False: #不是全连接，则增加连接
                      addConnections(net)
                   else: #已经是全连接，则对权值重新初始化
                      net.weightMatrix = np.random.uniform(-5, 5, (net.N + 1, m + net.N + 1))
                      for i in range(net.N + 1):
                         for j in range(m + net.N):
                            if net.connectedMatrix[i][j] == 0:
                               net.weightMatrix[i][j + 1] = 0
                            if j >= i + m:
                               net.connectedMatrix[i][j] = 0
                               net.weightMatrix[i][j + 1] = 0
                else: #否则减少结点个数
                    reduceNodenum(net)
        generation += 1
    print(generation)
