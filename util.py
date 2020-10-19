import gzip
import json
import datetime

def parse(path):
    """
    parsing the dataset
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
        
def convertTime(unix):
    return datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S')






