
import xgboost as xgb
import pandas as pd
import cPickle as pickle

params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.03
params["min_child_weight"] = 7
params["subsample"] = 0.7
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 10
plst = list(params.items())
data = pd.read_csv('final_data.csv')
y = data['Label']
X = data.drop(['Q_ID','U_ID','Label'],axis=1)
validate = pd.read_csv('new_validate_final.csv')
test = validate.drop(['qid','uid','label'],axis=1)
X = X.values
y = y.values
test = test.values
         
xgtrain = xgb.DMatrix(X,label=y)
xgtest = xgb.DMatrix(test)
             
num_rounds = 3000
model = xgb.train(plst,xgtrain,num_rounds)
prediction = model.predict(xgtest)

f = open('xgb_3000_0.03_10.pkl','wb')
pickle.dump(model,f)

prediction_dataFrame = pd.DataFrame(prediction)
prediction_dataFrame.to_csv('xgb_3000_0.03_10.csv')
