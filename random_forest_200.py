
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle as pickle

clf = RandomForestClassifier(n_estimators=200)
X = pd.read_csv('Xtransformed.csv')
Y = pd.read_csv('Y.csv')
X = X.values
Y = Y.values

X_test = pd.read_csv('Xtransformed_test.csv')
X_test = X_test.values

clf.fit(X, Y)

clf_file = open('clf_random_forest_200.pkl', 'wb')
pickle.dump(clf, clf_file)

prediction_random_forest_200 = clf.predict(X_test)

