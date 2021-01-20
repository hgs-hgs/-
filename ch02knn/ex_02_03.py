#ex_02_03.py手写数字识别
from numpy import *
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	#按距离升序排序，返回下标
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k): #距离最小的k个点
		VoteIlabel = labels[sortedDistIndicies[i]]
		classCount[VoteIlabel] = classCount.get(VoteIlabel,0)+1
	sortedClassCount = sorted(classCount.items(),
		key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def img2Vector(filename):
	returnVect = zeros((32,32))
	lines = [list(line.strip()) for line in open(filename)]
	returnVect[:] = lines[:]
	'''
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[i,j] = int(lineStr[j])
	''' 
	print(returnVect[0])
	return returnVect.reshape(1,1024)

def handWritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2Vector('trainingDigits/%s'%fileNameStr)

	testFileList = listdir('testDigits')
	errorCount = 0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2Vector('testDigits/%s' %fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("classifierResult is: %d the real answer is: %d "%(classifierResult,classNumStr))
		if classifierResult!=classNumStr:
			errorCount += 1.0

	print("\nthe total number of errors is: %d" %errorCount)
	print("\nthe total error rate is: %f" %(errorCount/float(mTest)))

if __name__ == '__main__':
	handWritingClassTest()