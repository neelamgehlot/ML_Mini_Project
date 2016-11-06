
from sklearn.decomposition import IncrementalPCA
import cPickle as pickle
import numpy as np
import pandas as pd

chunksize_ = 6000
dimensions = 300

reader = pd.read_csv('final_data.csv', sep = ',',chunksize = chunksize_)
#X = reader.drop(['Q_ID','U_ID','Label'], axis = 1)
#Y = reader['Label']

sklearn_pca = IncrementalPCA(n_components=dimensions)
for chunk in reader:
    chunk.pop('Q_ID')
    chunk.pop('U_ID')
    chunk.pop('Label')
    sklearn_pca.partial_fit(chunk)


# Computed mean per feature
mean = sklearn_pca.mean_
# and stddev
stddev = np.sqrt(sklearn_pca.var_)

reader = pd.read_csv('final_data.csv', sep = ',',chunksize = chunksize_)

sklearn_pca_file = open('sklearn_pca.pkl', 'wb')

pickle.dump(sklearn_pca, sklearn_pca_file)

Xtransformed = None
for chunk in reader:
    chunk.pop('Q_ID')
    chunk.pop('U_ID')
    chunk.pop('Label')
    Xchunk = sklearn_pca.transform(chunk)
    if Xtransformed == None:
        Xtransformed = Xchunk
    else:
        Xtransformed = np.vstack((Xtransformed, Xchunk))
        
Xtransformed_DataFrame = pd.DataFrame(Xtransformed)
Xtransformed_DataFrame.to_csv('Xtransformed.csv', index = False)

