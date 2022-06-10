# standard libraries
import numpy as np
import pandas as pd

# modelling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ndcg_score
from sklearn import metrics

# read the data
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
train.groupby('#QueryID')

# split and prepare the data to train 
train_df, vali_df = train_test_split(train, random_state=4, shuffle=False)
query_tr = train_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
X_train = train_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
y_train = train_df["Label"]
query_vali = vali_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
valiq = vali_df['#QueryID']
valid = vali_df['Docid']
X_vali = vali_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
y_vali = vali_df["Label"]
query_test = test.groupby('#QueryID')['#QueryID'].count().to_numpy()
testq = test['#QueryID']
testd = test['Docid']
X_test = test.drop(['#QueryID', 'Docid'], axis=1)

def run_paramater_sweep():
        group_info = query_tr.astype(int)
        flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
        gkf = GroupKFold(n_splits=10) 

        cv = gkf.split(X_train, y_train, groups=flatted_group)
        cv_group = gkf.split(X_train, groups=flatted_group)  # separate CV generator for manual splitting groups
        clf = lgb.LGBMRanker(objective='lambdarank', metric='ndcg')

        hyper_params = [{'learning_rate': [0.01,0.2],
                        'objective': ['lambdarank'],
                        'n_estimators': [40,60,100],
                        'boosting_type' : ['dart','gbdt'],
                        'num_leaves': [64,100], 
                        'random_state' : [2],
                        'colsample_bytree' : [0.4,0.5,0.65], # more if less feature vectors
                        'subsample' : [0.5,0.75] }]

        fit_params = {'eval_set': [(X_vali, y_vali)],
                    'eval_group': [query_vali],
                    'eval_metric': 'ndcg',
                    'early_stopping_rounds': 100,
                    'eval_at': [1]}

        # generator produces `group` argument for each fold
        def group_gen(flatted_group, cv):
            for train, _ in cv:
                yield np.unique(flatted_group[train], return_counts=True)[1]

        gen = group_gen(flatted_group, cv_group)
        grid = RandomizedSearchCV(clf, hyper_params, n_iter=10, cv=cv, verbose=2, scoring=ndcg_score, refit=False, random_state=None)

        grid.fit(X_train, y_train, group=next(gen), **fit_params)
        parameters = grid.best_params_
        return parameters
        '''
        Best Parameters:
        LGBMRanker(boosting_type='dart', colsample_bytree=0.4, num_leaves=64, objective='lambdarank', 
        random_state=2, subsample=0.5)
        '''
if __name__ == "__main__":

    sweep = True

    if sweep == True:

        model = lgb.LGBMRanker(  
        subsample= 0.5,
        random_state= 2,
        objective= 'lambdarank',
        num_leaves= 64,
        # n_estimators= 100,
        # learning_rate= 0.01,
        # importance_type= 'split',
        colsample_bytree= 0.4,
        boosting_type= 'dart') 

        params_fit = {  'eval_set': [(X_vali, y_vali)],
                        'eval_group': [query_vali],
                        'eval_metric': 'ndcg',
                        'early_stopping_rounds': 100,
                        'eval_at': [1] }

        model.fit(X_train, y_train, group=query_tr, **params_fit)
        prediction = model.predict(X_test, raw_score=False)

        scoredf = pd.DataFrame(columns=['QueryId', 'Docid', 'Score'])
        scoredf['QueryId'] = testq
        scoredf['Docid'] = testd
        scoredf['Score'] = prediction
        scoredf.to_csv("A2.run", sep="\t", header=False, index=False)

    else:

        parameters = run_paramater_sweep()  