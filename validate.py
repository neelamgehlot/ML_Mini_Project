
from sklearn.decomposition import IncrementalPCA
import cPickle as pickle
import numpy as np
import pandas as pd

chunksize_ = 6000
dimensions = 300

reader = pd.read_csv('validate_final.csv', sep = ',',chunksize = chunksize_)

sklearn_pca_file = open('sklearn_pca.pkl', 'rb')
sklearn_pca = pickle.load(sklearn_pca_file)

Xtransformed = None
for chunk in reader:
    chunk.pop('qid')
    chunk.pop('uid')
    chunk.pop('label')
    Xchunk = sklearn_pca.transform(chunk)
    if Xtransformed == None:
        Xtransformed = Xchunk
    else:
        Xtransformed = np.vstack((Xtransformed, Xchunk))
        
Xtransformed_DataFrame = pd.DataFrame(Xtransformed)
Xtransformed_DataFrame.to_csv('Xtransformed_test.csv', index = False)
        
