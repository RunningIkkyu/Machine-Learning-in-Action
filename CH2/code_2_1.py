import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1))
    sqMat = diffMat ** 2
    sqDis = sqMat.sum(axis=1)
    distance = sqDis ** 0.5
    sortedDis = distance.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDis[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    data = np.array([[1.0, 1.0], [1.0, 1.1], [0, 0], [0.1, 0]])
    labels = ['A', 'A', 'B', 'B']
    inX = [1.2, 1.2]
    k = 3
    classify0(inX, data, labels, k)