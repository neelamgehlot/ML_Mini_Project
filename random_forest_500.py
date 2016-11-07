
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle as pickle

clf = RandomForestClassifier(n_estimators=500)
X = pd.read_csv('Xtransformed.csv')
Y = pd.read_csv('Y.csv')
X = X.values
Y = Y.values

X_test = pd.read_csv('Xtransformed_test.csv')
X_test = X_test.values

clf.fit(X, Y)

clf_file = open('clf_random_forest_500.pkl', 'wb')
pickle.dump(clf, clf_file)

prediction_random_forest_500 = clf.predict_proba(X_test)
pred = pd.DataFrame(prediction_random_forest_500)

pred.to_csv('prediction_random_forest_500.csv', index = False)

validate = pd.read_csv('new_validate_final.csv')
v_ids = validate[['qid','uid']]

concatinated = pd.concat([v_ids, pred], axis = 1)
validate_nolabel = pd.read_csv('validate_nolabel.txt')
result = pd.merge(validate_nolabel, concatinated, how="left", on=['qid','uid']

result.to_csv('prediction.csv', index = False)


