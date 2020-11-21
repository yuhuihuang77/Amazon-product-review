import gzip
import json
import datetime
import csv, re
import numpy as np

def parse(path):
    """
    parsing the dataset
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

        
def convertTime(unix):
    return datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S')


def preprocessData(path):
    '''
    unixReviewTime = 0
    reviewTime = 1
    ratings = 2
    reviewText = 3
    reviewerID = 4
    productID = 5
    '''
    ratings = []
    reviewText = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            ratings.append(int(row[2]))
            s = row[3]
            # replace images with the word 'image'
            s = re.sub(r'<div.*>&nbsp;', ' image ', s)
            # replace dollar signs with 'dollar'
            s = re.sub(r'$', ' dollar ', s)
            # replace numbers with 'number'
            s = re.sub(r'\d+', ' number ', s)
            # remove punctuations
            s = re.sub(r'[^\w]', ' ', s)
            # replace 
            s = s.lower()
            reviewText.append(s)
    return ratings, reviewText


def createFeatures(reviews, ratings, wordToIndex, m):
    n = len(wordToIndex)
    y = np.array(ratings[:m])
    X = np.zeros((m, n), dtype = np.bool_)
    for i in range(m):
        for word in reviews[i].split():
            if word in wordToIndex:
                X[i, wordToIndex[word]] = 1
    return X, y

def createWordVectorFeatures(reviews, ratings, dic, m, n = 300):
    y = np.array(ratings[:m])
    X = np.zeros((m, n))
    for i in range(m):
        for word in reviews[i].split():
            if word in dic:
                X[i, :] += dic[word]
    return X, y
                
    
def linRegPredict(model, Xval):
    yPredict = model.predict(Xval)
    yPredict = np.rint(yPredict).astype(np.int)
    yPredict[yPredict < 1] = 1
    yPredict[yPredict > 5] = 5
    return yPredict

def evalModel(yPred, yval):
    accuracy = np.mean((yPred == np.array(yval)))
    confM = np.zeros((5, 5))
    for i in range(len(yval)):
        confM[yval[i]-1][yPred[i]-1] += 1
    # compute the average F1 score
    pred = np.sum(confM, axis = 0)
    val = np.sum(confM, axis = 1)
    precision = np.array([confM[i][i] / pred[i] if pred[i] > 0 else 0 for i in range(5)])
    recall = np.array([confM[i][i] / val[i] if val[i] > 0 else 0 for i in range(5)])
    denom = precision + recall
    denom[denom == 0] = 1
    F1 = 2 * precision * recall / denom
    F1[np.isnan(F1)] = 0
    return accuracy, np.mean(F1), confM
    


