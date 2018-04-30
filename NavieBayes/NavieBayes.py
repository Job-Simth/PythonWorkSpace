# -*- coding: UTF-8 -*-
'''
Author: marco lin
Date: 2015-08-28
'''

from numpy import *
import pickle
import jieba
import time

stop_word = ['，', '。', '、', '！', '？', ',', '.', '!', '?', ' ', '', '\n', '（', '）', '(', ')', '\ufeff']
'''
    停用词集, 包含“啊，吗，嗯”一类的无实意词汇以及标点符号
'''

'''
    载入数据
'''


def loadStopword():
    fr = open('stopword.txt', 'r', encoding=('utf-8'))
    lines = fr.readlines()
    for line in lines:
        stop_word.append(line.strip())
    fr.close()


'''
    创建词集
    params:
        documentSet 为训练文档集
    return:词集, 作为词袋空间
'''


def createVocabList(documentSet):
    vocabSet = set([])
    for document in documentSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


'''
   文本处理，如果是未处理文本，则先分词（jieba分词）,再去除停用词
'''


def textParse(bigString):  # input is big string, #output is word list
    cutted = jieba.cut(bigString, cut_all=False)
    listOfWord = []
    for word in cutted:
        if word not in stop_word:
            listOfWord.append(word)
    return listOfWord


'''
    交叉训练
'''
BAD = 1
GOOD = 0


def testClassify():
    listAllDoc = []
    listClasses = []

    print("----loading document list----")

    # 31个标注为差评的文档
    for i in range(1, 32):
        wordList = textParse(open('bad/%d.txt' % i, 'r', encoding=('utf-8')).read())
        listAllDoc.append(wordList)
        listClasses.append(BAD)
    # 31个标注为好评的文档
    for i in range(1, 32):
        wordList = textParse(open('good/%d.txt' % i, 'r', encoding=('utf-8')).read())
        listAllDoc.append(wordList)
        listClasses.append(GOOD)

    print("----creating vocab list----")
    # 构建词袋模型
    listVocab = createVocabList(listAllDoc)
    docNum = len(listAllDoc)
    # testSetNum = int(docNum * 0.1)
    testSetNum = 10

    trainingIndexSet = list(range(docNum))  # 建立与所有文档等长的空数据集（索引）
    testSet = []  # 空测试集

    # 随机索引，用作测试集, 同时将随机的索引从训练集中剔除
    for i in range(testSetNum):
        randIndex = int(random.uniform(0, len(trainingIndexSet)))
        testSet.append(trainingIndexSet[randIndex])
        del (trainingIndexSet[randIndex])

    trainMatrix = []
    trainClasses = []

    for docIndex in trainingIndexSet:
        trainMatrix.append(bagOfWords2VecMN(listVocab, listAllDoc[docIndex]))
        trainClasses.append(listClasses[docIndex])

    print("----traning begin----")
    pBABV, pGOODV, pCLASS = trainNaiveBayes(array(trainMatrix), array(trainClasses))

    print("----traning complete----")
    print("pBABV:", pBABV)
    print("pGOODV:", pGOODV)
    print("pCLASS:", pCLASS)
    print("bad: %d, good:%d" % (BAD, GOOD))

    args = dict()
    args['pBABV'] = pBABV
    args['pGOODV'] = pGOODV
    args['pCLASS'] = pCLASS

    fw = open("args.pkl", "wb")
    pickle.dump(args, fw, 2)
    fw.close()

    fw = open("vocab.pkl", "wb")
    pickle.dump(listVocab, fw, 2)
    fw.close()

    errorCount = 0
    for docIndex in testSet:
        vecWord = bagOfWords2VecMN(listVocab, listAllDoc[docIndex])
        if classifyNaiveBayes(array(vecWord), pBABV, pGOODV, pCLASS) != listClasses[docIndex]:
            errorCount += 1
            doc = ' '.join(listAllDoc[docIndex])
            print("classfication error", doc)
    print('the error rate is: ', float(errorCount) / len(testSet))


# 分类方法(这边只做二类处理)
def classifyNaiveBayes(vec2Classify, pBADVec, pGOODVec, pClass1):
    pIsBAD = sum(vec2Classify * pBADVec) + log(pClass1)  # element-wise mult
    pIsGOOD = sum(vec2Classify * pGOODVec) + log(1.0 - pClass1)

    if pIsBAD > pIsGOOD:
        return BAD
    else:
        return GOOD


'''
    训练
    params:
        tranMatrix 由测试文档转化成的词空间向量 所组成的 测试矩阵
        tranClasses 上述测试文档对应的分类标签
'''


def trainNaiveBayes(trainMatrix, trainClasses):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # 计算矩阵列数, 等于每个向量的维数
    numIsBAD = len([x for x in trainClasses if x == BAD])
    pCLASS = numIsBAD / float(numTrainDocs)
    pBADNum = ones(numWords)
    pGOODNum = ones(numWords)
    pBADDemon = 2.0
    pGOODDemon = 2.0

    for i in range(numTrainDocs):
        if trainClasses[i] == BAD:
            pBADNum += trainMatrix[i]
            pBADDemon += sum(trainMatrix[i])
        else:
            pGOODNum += trainMatrix[i]
            pGOODDemon += sum(trainMatrix[i])

    pBADVect = log(pBADNum / pBADDemon)
    pGOODVect = log(pGOODNum / pGOODDemon)

    return pBADVect, pGOODVect, pCLASS


'''
    将输入转化为向量，其所在空间维度为 len(listVocab)
    params: 
        listVocab-词集
        inputSet-分词后的文本，存储于set
'''


def bagOfWords2VecMN(listVocab, inputSet):
    returnVec = [0] * len(listVocab)
    for word in inputSet:
        if word in listVocab:
            returnVec[listVocab.index(word)] += 1
    return returnVec


'''
    读取保存的模型，做分类操作
'''


def Classify(textList):
    fr = open("args.pkl", "rb")
    args = pickle.load(fr)
    pBABV = args['pBABV']
    pGOODV = args['pGOODV']
    pCLASS = args['pCLASS']
    fr.close()

    fr = open("vocab.pkl", "rb")
    listVocab = pickle.load(fr)
    fr.close()

    if len(listVocab) == 0:
        print("got no args")
        return

    text = textParse(textList)
    vecWord = bagOfWords2VecMN(listVocab, text)
    class_type = classifyNaiveBayes(array(vecWord), pBABV, pGOODV, pCLASS)
    if class_type == 1:
        print("classfication type:差评")
        return BAD
    else:
        print("classfication type:好评")
        return GOOD


if __name__ == "__main__":
    loadStopword()
    goodCount = 0
    badCount = 0
    while True:
        opcode = input("input 1 for training, 2 for classify, 3 for test: ")
        if opcode.strip() == "1":
            begtime = time.time()
            testClassify()
            print("cost time total:", time.time() - begtime)
        elif opcode.strip() == "2":
            textList = open('taobao.txt', 'r', encoding=('utf-8')).readlines()
            print(len(textList))
            for text in textList:
                if Classify(text) == BAD:
                    badCount += 1
                else:
                    goodCount += 1
            print(goodCount)
            print(badCount)
            print("好评率：", goodCount / (goodCount + badCount))
            goodCount = 0
            badCount = 0
        else:
            text = input("input the text:")
            Classify(text)
