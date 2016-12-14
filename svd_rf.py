from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle

users = pd.read_csv('user_with_word_tags.csv')
users.columns = ['uid'] + list(users.columns[1:])
users_with_svd = users.ix[:, 1:]
svd_u = TruncatedSVD(n_components=5000)
svd_u.fit(users_with_svd)
users_transform = svd_u.transform(users_with_svd)
users_transform = pd.DataFrame(users_transform)

users_transform.to_csv('users_transform_svd_5000.csv', index = False)
final_users_data = pd.concat([users.ix[:, 0],users_transform],axis=1)
final_users_data.to_csv('users_data_svd_5000.csv', index = False)

####

ques = pd.read_csv('ques_with_word_tags.csv')
ques.columns = ['qid', 'qtag', 'upvotes', 'ans', 'top_ans'] + list(ques.columns[5:])
ques_without_svd = ques.ix[:, 1:5]
ques_with_svd = ques.ix[:, 5:]
svd_q = TruncatedSVD(n_components=4000)
svd_q.fit(ques_with_svd)
ques_transform = svd_q.transform(ques_with_svd)
ques_transform = pd.DataFrame(ques_transform)
ques_transform = pd.concat([ques_without_svd,ques_transform],axis=1)

ques_transform.to_csv('ques_transform_svd_4000.csv', index = False)
final_ques_data = pd.concat([ques.ix[:, 0],ques_transform],axis=1)
final_ques_data.to_csv('ques_data_svd_4000.csv', index = False)

####

invited = pd.read_csv('invited_info_train.txt', delim_whitespace = True, header = None)
invited.columns = ['qid', 'uid', 'label']

ques_merge = pd.merge(invited, final_ques_data, on = 'qid')
train = pd.merge(ques_merge, final_users_data, on='uid')

train.to_csv('train_data_1.csv', index = False)

###

validate_nolabel = pd.read_csv('validate_nolabel.txt')

ques_merge_validate = pd.merge(validate_nolabel, final_ques_data, on = 'qid')
validate = pd.merge(ques_merge_validate, final_users_data, on='uid')

validate.to_csv('validate_data_1.csv', index = False)

###

clf = RandomForestClassifier(n_estimators=1000)
X = pd.read_csv('train_data_1.csv')
X = X.ix[:, 3:]
Y = pd.read_csv('Y.csv')
X = X.values
Y = Y.values

X_test = pd.read_csv('validate_data_1.csv')
v_ids = X_test.ix[:, :2]
X_test = X_test.ix[:, 3:]
X_test = X_test.values

clf.fit(X, Y)

clf_file = open('clf_RF_1000_SVD.pkl', 'wb')
pickle.dump(clf, clf_file)

pred_RF_1000_SVD = clf.predict_proba(X_test)
pred = pd.DataFrame(pred_RF_1000_SVD)

pred.to_csv('pred_RF_1000_SVD.csv', index = False)


concatinated = pd.concat([v_ids, pred], axis = 1)
validate_nolabel = pd.read_csv('validate_nolabel.txt')
result = pd.merge(validate_nolabel, concatinated, how="left", on=['qid','uid']

result.to_csv('prediction.csv', index = False)

































train = pd.read_csv('final_data.csv')
x_without_svd = train.ix[:,3:7]
x_with_svd = train.ix[:,7:]
svd = TruncatedSVD(n_components=500)
svd.fit(x_with_svd)
x_transform = svd.transform(x_with_svd)
x_transform = pd.DataFrame(x_transform)
final_x = pd.concat([x_without_svd,x_transform],axis=1)
final_x.to_csv('Xtrain_SVD_500.csv',index=False)

validate = pd.read_csv('new_validate_final.csv')
v_without_svd = validate.ix[:,3:7]
v_with_svd = validate.ix[:,7:]
v_transform = svd.transform(v_with_svd)
v_transform = pd.DataFrame(v_transform)
final_v = pd.concat([v_without_svd,v_transform],axis=1)
final_v.to_csv('Xtest_SVD_500.csv',index=False)

X_test = pd.read_csv('Xtest_SVD_300.csv')
