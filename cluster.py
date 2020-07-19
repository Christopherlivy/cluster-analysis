import numpy as np
import matplotlib.pyplot as plt
import random as rd
dotcolor=['.r','.g','.y','.b','.m','.k']		#点的颜色
labelcolor=['*r','*g','*y','*b','*m','*k']		#簇心的颜色
def readDtata():
	data=[]
	with open('./cluster.dat','r') as f:	#以可读方式打开dat数据文件
		for line in f.readlines():		#逐行读入数据
			line=list(line.strip().split())
			line[0]=eval(line[0])
			line[1]=eval(line[1])
			current=[]
			for i in line:
				current.append(i)	#存入current作为二维数组
			data.append(current)
		TestData=rd.sample(data,200)	#随机选择200个作为测试数据
		TrainData=[i for i in data if i not in TestData]	#将剩下的800个作为训练数据
		return np.mat(TestData),np.mat(TrainData)
def drawpicture(input,center,postion,number,k):		#画出散点图
	for i in range(number):
		if postion[i]==-1:
			postion[i]=5	#初始时将点画为黑色
		plt.plot(input[i,0],input[i,1],dotcolor[postion[i]])	#画每个点
	for i in range(k):		#画簇心
		plt.plot(center[i,0],center[i,1],labelcolor[i])
	plt.show()

def BetweenDistance(m,n):	#计算两个点的距离
	length=len(m)
	total=0
	for temp in range(2):	#通过分别计算每个维度来计算两个点的欧式距离
		distance=(m[0,temp]-n[0,temp])**2
		total+=distance
	return np.sqrt(total)




def getInitialCenter(input,k):	#初始化质心
	center=np.mat(np.zeros((k,2)))	#得到质心的矩阵
	for i in range(2):
		r=np.random.rand(k,1)*(max(input[:,i])-min(input[:,i]))	#设置随机因子
		center[:,i]=(min(input[:,i])+r)	#随机初始化
	return center

def getNewCenter(input,postion,k,number):	#更新簇心位置
	newcenter=np.zeros((k,2))
	for i in range(k):		#对于所有的簇心
		n=0
		temp = []
		for j in range(number):	#对于所有属于簇心i的那些点，加入到temp数组中
			if postion[j]==i:
				n+=1
				temp.append(input[j,:])
		if n!=0:
			newcenter[i,:]=np.mean(temp,axis=0)	#如果簇心不为空，通过属于簇心的点来更新簇心
	return np.mat(newcenter)




def KMeansClusterAnalysis(TestData,TrainData,k,flag=1):
	number=np.shape(TrainData)[0]	#得到点的数量
	center=getInitialCenter(TrainData,k)	#得到k个初始点
	train_postion=[-1]*number	#所有点的归属的簇心
	test_postion=[-1]*200
	epoch_loss=[]
	drawpicture(TrainData, center, train_postion, 800, k)	#画出初始点
	while flag==1:
		min_distance = [9999] * number	#所有点到某个簇心的最小距离
		sum_loss=0
		for i in range(200):	#损失等于所有点的损失和
			signal_loss=9999
			for j in range(k):	#计算每个点的损失
				distance=BetweenDistance(TestData[i],center[j])
				if distance<signal_loss:
					signal_loss=distance
					test_postion[i]=j
			sum_loss+=signal_loss
		epoch_loss.append(sum_loss)	#得到这一代的损失SSE
		flag=0
		for i in range(number):	#对于所有点
			for j in range(k): 	 #对于所有簇心
				distance=BetweenDistance(TrainData[i],center[j])	#计算两点距离
				if distance < min_distance[i]:	#如果距离小于记录的最小距离
					min_distance[i]=distance	#更新最小距离
					new_postion =j	#更新属于的簇心
			#print(postion[i],new_postion)
			if train_postion[i]!= new_postion:	#如果有点的簇心发生变化
				flag=1	#将继续迭代标志置为有效
			train_postion[i]=new_postion	#更新归属的簇心
		center=getNewCenter(TrainData,train_postion,k,number)	#更新簇心
	drawpicture(TrainData, center, train_postion, 800, k)	#画出最终训练集的散点图
	drawpicture(TestData, center, test_postion, 200, k)		#画出测试集的散点图
	print(epoch_loss)
	plt.xlabel('k')	#横坐标
	plt.ylabel('SSE LOST')#纵坐标
	plt.plot(epoch_loss,color='y',marker='*',lw=1,ms=5,mfc='w',mec='r')	#画出折线图
	for i in range(len(epoch_loss)):	#画出折线图的坐标点数值
		plt.text(i,epoch_loss[i],int(epoch_loss[i]),ha='center',va='bottom',fontsize='7')
	plt.show()
	return min_distance



if __name__=="__main__":
	TestData,TrainData=readDtata()
	KMeansClusterAnalysis(TestData,TrainData,3)
