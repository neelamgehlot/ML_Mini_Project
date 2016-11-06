
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle as pickle

clf = RandomForestClassifier(n_estimators=200)
X = pd.read_csv('final_data.csv')
Y = X['Label']
X = X.values
Y = Y.values

X_test = pd.read_csv('validate_final.csv')
X_test = X_test.values

clf.fit(X, Y)

#clf_file = open('clf_random_forest_200.pkl', 'wb')
#pickle.dump(clf, clf_file)

prediction_random_forest_200 = clf.predict(X_test)
pred = pd.DataFrame(prediction_random_forest_200)
pred.to_csv('random_forest_200_full_data.csv', index = False)

validate = pd.read_csv('validate_nolabel.txt')

validate['label'] = pred
validate.to_csv('prediction_full_data.csv', index = False)
