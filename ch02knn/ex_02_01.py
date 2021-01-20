from numpy import *
import operator

def createDataSet():
	group = array([ [1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX, dataSet, labels, k):
	#inX测试实例
	#dataSet训练集
	
	[dataSetSize,featureSize] = dataSet.shape #训练集实例个数,特征维度
	
	#在矩阵内完成最小二乘法，即计算距离
	diffMat = tile(inX, (dataSetSize,1)) #将inX复制扩展，行方向dataSetSize次，列方向1次
	diffMat -= dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5

	#按距离升序排序，返回原下标
	sortedDistIndicies = distances.argsort()

	#选取距离最小的k个点, 对其中标签进行分类统计 
	classCount = {}
	for i in range(k): 
		VoteIlabel = labels[sortedDistIndicies[i]]
		classCount[VoteIlabel] = classCount.get(VoteIlabel,0)+1
		
	sortedClassCount = sorted(classCount.items(),
		key=operator.itemgetter(1),reverse=True)

	return sortedClassCount[0][0]


if __name__=='__main__':
	dataSet,labels = createDataSet()
	testSet = [[1.1, 1.0], [0.2, 0.3], [0.1, 0.2], [0.9, 0.8]]
	k = 3
	for inX in testSet:
		print('inX',inX,'-----> class',classify0(inX, dataSet, labels,k))
	
	
