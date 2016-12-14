from sklearn.decomposition import TruncatedSVD
import pandas as pd
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
