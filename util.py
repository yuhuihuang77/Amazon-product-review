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
            reviewText.append(s.lower())
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
                


