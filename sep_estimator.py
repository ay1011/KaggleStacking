import pandas as pd
import xgboost as xgb

class sep_estimator:
    def __init__(self,params,cols,estimators):
        self.est={}
        self.cols=cols
        for j in range(len(self.cols)):
            self.est[j]=estimators[j](**params[j])

    def preprocess(self,data, label_value, mode,):
        if mode=='predict':
            assert (data['labels'].values==label_value).all()
        data = data.loc[:,self.cols[label_value]]
        return data

    def fit(self,X,y):
        for j in range(len(self.cols)):
            Xc = X.copy()
            Xc = self.preprocess(Xc,j,'train')
            self.est[j].fit(Xc,y)

    def predict(self, X):
        res = pd.DataFrame(index=X.index)
        for j in range(len(self.cols)):
            Xc = X.loc[X['labels'] == j, :].copy()
            Xtemp = self.preprocess(Xc, j, 'predict')
            if len(Xtemp) > 0:
                pred = self.est[j].predict(Xtemp)
                res.loc[Xc.index, 0] = pred

        return res[0].values.flatten()
