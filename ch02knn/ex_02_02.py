#ex_02_02.py
from numpy import * 
import matplotlib
import matplotlib.pyplot as plt
import operator

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	#计算距离矩阵，并排序
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	#选前k个实例，对标签做分类统计
	classCount = {}
	for i in range(k): 
		VoteIlabel = labels[sortedDistIndicies[i]]
		classCount[VoteIlabel] = classCount.get(VoteIlabel,0)+1
	sortedClassCount = sorted(classCount.items(),
		key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

#从文件读取数据
def file2matrix(filename):
	lines = [line.strip().split('\t') for line in open(filename)]
	dataSet = zeros((len(lines),3))
	classLabelVector = []
	index = 0
	for line in lines:
		dataSet[index] = line[0:3]#强制转化为数值
		classLabelVector.append(int(line[-1]))
		index += 1
	return dataSet,classLabelVector

#归一化数据集
def autoNorm(dataSet):
	minval = dataSet.min(0)
	maxval = dataSet.max(0)
	ranges = maxval - minval
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet-tile(minval,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minval

#算法测试
def datingClassTest():
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minval = autoNorm(datingDataMat)

	m = normMat.shape[0]
	hoRatio = 0.10 #90%做训练集，10%测试集
	numTestVecs = int(m*hoRatio)

	errorCount = 0
	for i in range(numTestVecs):
		calssifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m], 3)
		print("The calssifier came back with: %d, the real answer is: %d" 
			%(calssifierResult,datingLabels[i]))
		if calssifierResult != datingLabels[i]:
			errorCount += 1.0
	print("The total error rate is: %f%%" %(errorCount/float(numTestVecs)*100))

#构建完整系统
def classfiPerson():
	resultList = ['Not at all', 'In small doses', 'In large doses']
	ffMiles = float(input("frequent flier miles earned per year?"))
	percenTals = float(input("Percentage of time spent playing video games?"))
	iceCream = float(input("Liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minval = autoNorm(datingDataMat)
	inArr =array([ffMiles,percenTals,iceCream])
	calssifierResult = classify0((inArr - minval)/ranges, normMat, datingLabels,3)
	print("You will probably like this person: ",resultList[calssifierResult-1])

if __name__ == '__main__':
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	datingClassTest()
	classfiPerson()
	'''
	fig = plt.figure()
	ax =fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
		15.0*array(datingLabels),15.0*array(datingLabels))
	plt.show()
	'''