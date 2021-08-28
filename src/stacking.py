import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from scipy.optimize import minimize, fmin
from xgboost import XGBClassifier
from sklearn import model_selection, linear_model, metrics, decomposition, ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone

from tqdm import tqdm
from functools import partial, reduce
from typing import List
import warnings
import random
import glob
import os

warnings.simplefilter(action='ignore')
NUM_FOLDS = 5

class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """Linear regression for model-based AUC optimization.
    Note that we transform probabilities to rank probabilities!"""
    
    def __init__(self): 
        self.lr = linear_model.LinearRegression()
        
    def fit(self, X, y):
        self.lr.fit(pd.DataFrame(X).rank(), y)
        return self
        
    def predict_proba(self, X):
        return np.c_[[0]*len(X), self.lr.predict(pd.DataFrame(X).rank())]


class ReviewColumnExtractor(BaseEstimator, ClassifierMixin):
    """Extract text column, e.g. letting X = df_train[['review']]
    as train dataset for TfidfVectorizer and CountVectorizer does
    not work as expected."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.review


class StackingClassifierParallel(BaseEstimator, ClassifierMixin):
    """Implements model stacking for classification."""
    
    def __init__(self, model_dict_list, n_jobs=1):
        """Initialize by passing `model_dict` which is a list of dictionaries 
        of name-model pairs for each level."""
        
        self.model_dict_list = [
            {name : clone(model_dict[name]) for name in model_dict} \
                                            for model_dict in model_dict_list
        ]
        self.cv_scores_ = {}
        self.metafeatures_ = None
        self.n_jobs = n_jobs
        
    def fit(self, df):
        """Fit classifier. Assumes `df` is a DataFrame with 'kfold' and 
        'sentiment' (target) columns, followed by features columns."""
        
        # Iterating over all stacking levels
        df = df.copy()
        metafeatures = []
        for m in tqdm(range(len(self.model_dict_list))):
            
            # Get models in current layer
            model_dict = self.model_dict_list[m]
            level = m + 1
            
            # Identify feature columns, i.e. preds of prev. layer
            if m == 0:
                feature_cols = ['review']
            else:
                prev_level_names = self.model_dict_list[m-1].keys()
                feature_cols = [f'{name}_{level-1}' for name in prev_level_names]
            
            # Iterate over models in the current layer
            with Parallel(n_jobs=self.n_jobs, backend='loky', verbose=2) as parallel:
                print(df.head())
                for model_name in tqdm(model_dict.keys()):
                    model = model_dict[model_name]
                    
                    # Generate feature for next layer models from OOF preds
                    model = model_dict[model_name]
                    out = parallel(delayed(self._predict_fold)(
                            df,
                            feature_cols,
                            fold,
                            model_name,
                            clone(model),
                            level
                        ) for fold in df.kfold.unique()
                    )

                    # Load all OOF predictions and AUCs
                    fold_preds, cv_scores = list(zip(*out))
                    
                    # Assign cv scores for model and append predictions to df
                    self.cv_scores_[f'{model_name}_{level}'] = cv_scores
                    pred_df = pd.concat(fold_preds)
                    df = df.merge(pred_df, how='left', on='id')
                    metafeatures.append(f'{model_name}_{level}')
                    
                    # Refit model on entire feature columns for inference
                    model.fit(df[feature_cols], df.sentiment)
        
        # Save learned metafeatures
        self.metafeatures_ = df[metafeatures]
        return self
    
    def predict_proba(self, df):
        """Return classification probabilities."""
        
        df = df.copy()
        
        # Iterate over layers to make predictions
        for m in range(len(self.model_dict_list)):
            
            # Get models for current layer
            model_dict = self.model_dict_list[m]
            level = m + 1
            
            # Get feature columns to use for prediction
            if m == 0:
                feature_cols = ['review']
            else:
                prev_names = self.model_dict_list[m-1].keys()
                feature_cols = [f"{model_name}_{level-1}" for model_name in prev_names]

            # Append predictions to test DataFrame
            for model_name in model_dict.keys():
                model = model_dict[model_name]
                pred = model.predict_proba(df[feature_cols])[:, 1] 
                df.loc[:, f"{model_name}_{level}"] = pred
                    
        # Return last predictions
        return np.c_[1 - pred, pred]

    def _predict_fold(self, df,
                            feature_cols,
                            fold,
                            model_name,
                            model, level):
        "Train on train, predict on valid. Return OOF predictions with AUC score."
        
        X_train = df[df.kfold != fold][feature_cols]
        X_valid = df[df.kfold == fold][feature_cols] 
        y_train = df[df.kfold != fold].sentiment.values
        y_valid = df[df.kfold == fold].sentiment.values
        X_valid_id = df[df.kfold == fold].id

        # Fit model
        model.fit(X_train, y_train)
        
        # Return fold predictions along with fold AUC
        pred = model.predict_proba(X_valid)[:, 1] 
        auc = metrics.roc_auc_score(y_valid, pred)
        return pd.DataFrame({"id": X_valid_id, f"{model_name}_{level}": pred}), auc


class Blender(BaseEstimator, ClassifierMixin):
    """Implement blending that maximizes AUC score."""
    
    def __init__(self, rank=False):
        self.coef_ = None
        self.rank = rank

    def fit(self, X, y):
        """Find optimal blending coefficients."""
        
        if self.rank:
            X = X.rank()

        self.coef_ = self._optimize_auc(X, y)
        return self

    def predict_proba(self, X):
        """Return blended probabilities for class 0 and class 1."""
        
        if self.rank:
            X = X.rank()
            
        pred = np.sum(X * self.coef_, axis=1)
        return np.c_[1 - pred, pred]

    def _auc(self, coef, X, y):
        """Calculate AUC of blended predict probas."""

        auc = metrics.roc_auc_score(y, np.sum(X * coef, axis=1))
        return -1.0 * auc # min -auc = max auc
    
    def _optimize_auc(self, X, y):
        """Maximize AUC as a bound-constrained optimization problem using Nelder-Mead 
        method with Dirichlet init. 
        
        Reference: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        partial_loss = partial(self._auc, X=X, y=y) 
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        return minimize(partial_loss, init_coef, 
                        method='Nelder-Mead', 
                        bounds=[(0, 1)]*X.shape[1])['x']


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """Implements model stacking for classification."""
    
    def __init__(self, model_dict_list):
        """Initialize by passing `model_dict` which is a list of dictionaries 
        of name-model pairs for each level."""
        
        self.model_dict_list = model_dict_list
        self.cv_scores_ = {}
        self.metafeatures_ = None
        
    def fit(self, df):
        """Fit classifier. This assumes `df` is a DataFrame with "id", "kfold", 
        "sentiment" (target) columns, followed by features columns."""
        
        df = df.copy()
        
        # Iterating over all stacking levels
        metafeatures = []
        for m in range(len(self.model_dict_list)):
            
            # Get models in current layer
            model_dict = self.model_dict_list[m]
            level = m + 1
            
            # Identify feature columns, i.e. preds of prev. layer
            if m == 0:
                feature_cols = ['review']
            else:
                prev_level_names = self.model_dict_list[m-1].keys()
                feature_cols = [f'{name}_{level-1}' for name in prev_level_names]
            
            # Iterate over models in the current layer
            for model_name in model_dict.keys():
                print(f'\nLevel {level} preds: {model_name}')
                self.cv_scores_[f'{model_name}_{level}'] = []
                model = model_dict[model_name]
                
                # Generate feature for next layer models from OOF preds
                fold_preds = []
                for j in range(df.kfold.nunique()):
                    fold_pred, fold_auc = self._predict_fold(df, feature_cols, model, 
                                                        model_name, fold=j, level=level)
                    fold_preds.append(fold_pred)
                    self.cv_scores_[f'{model_name}_{level}'].append(fold_auc)
                
                pred = pd.concat(fold_preds)
                df = df.merge(pred[['id', f'{model_name}_{level}']], on='id', how='left')   
                metafeatures.append(f'{model_name}_{level}')
        
                # Train models on entire feature columns for inference
                print(feature_cols)
                model.fit(df[feature_cols], df.sentiment.values)
        
        self.metafeatures_ = df[metafeatures]
        return self
        
    def predict_proba(self, df):
        """Return classification probabilities."""
        
        df = df.copy()
        
        # Iterate over layers to make predictions
        for m in range(len(self.model_dict_list)):
            
            # Get models for current layer
            model_dict = self.model_dict_list[m]
            level = m + 1
            
            # Get feature columns to use for prediction
            if m == 0:
                feature_cols = ['review']
            else:
                prev_names = self.model_dict_list[m-1].keys()
                feature_cols = [f"{model_name}_{level-1}" for model_name in prev_names]

            # Append predictions to test DataFrame
            print(feature_cols)
            for model_name in model_dict.keys():
                model = model_dict[model_name]
                pred = model.predict_proba(df[feature_cols])[:, 1] 
                df.loc[:, f"{model_name}_{level}"] = pred
                    
        # Return last predictions
        return np.c_[1 - pred, pred]
        
    def _predict_fold(self, df, feature_cols, model, model_name, fold, level):
        "Train on K-1 folds, predict on fold K. Return OOF predictions with IDs."

        # Get folds; include ID and target cols, and feature cols
        df_trn = df[df.kfold != fold][['id', 'sentiment']+feature_cols]
        df_oof = df[df.kfold == fold][['id', 'sentiment']+feature_cols]
        
        # Fit model. 
        model.fit(df_trn[feature_cols], df_trn.sentiment.values)
        fold_pred = model.predict_proba(df_oof[feature_cols])[:, 1] 
        auc = metrics.roc_auc_score(df_oof.sentiment.values, fold_pred)
        print(f"fold={fold}, auc={auc}")

        # Return OOF predictions with ids
        df_oof.loc[:, f"{model_name}_{level}"] = fold_pred
        return df_oof[["id", f"{model_name}_{level}"]], auc                     

if __name__ == '__main__':
    # Base models
    # Base models
    # Base models
    level1 = {
        'lr': make_pipeline(
            ReviewColumnExtractor(),
            TfidfVectorizer(max_features=1000),
            linear_model.LogisticRegression()
        ), 
        
        'lr_cnt': make_pipeline(
            ReviewColumnExtractor(),
            CountVectorizer(), 
            linear_model.LogisticRegression(solver='liblinear')
        ), 
    }

    # Meta models
    level2 = {
        'lr': linear_model.LogisticRegression(),
        'linreg': make_pipeline(StandardScaler(), LinearRegressionClassifier()),
        'xgb': XGBClassifier(eval_metric="logloss", use_label_encoder=False, nthread=1)
    }

    # Meta models
    level3 = {
        'linreg': make_pipeline(StandardScaler(), LinearRegressionClassifier()),
        'xgb': XGBClassifier(eval_metric="logloss", use_label_encoder=False, nthread=1)
    }

    # Blender head: rank true for linear reg.
    level4 = {'blender': Blender(rank=True)}

    df = pd.read_csv('../data/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv', sep='\t')
    df.head()

    df_train, df_test = model_selection.train_test_split(df, test_size=0.20)
    print(df_train.shape, df_test.shape)

    df_train.loc[:, 'kfold'] = -1 
    df_train = df_train.sample(frac=1.0).reset_index(drop=True)
    y = df_train['sentiment'].values

    skf = model_selection.StratifiedKFold(n_splits=NUM_FOLDS)
    for f, (t_, v_) in enumerate(skf.split(X=df_train, y=y)):
        df_train.loc[v_, "kfold"] = f

    import time

    model_dict_list = [level1, level3, level4]

    # parallel
    start = time.time()
    stack_parallel = StackingClassifierParallel(model_dict_list, n_jobs=-1)
    stack_parallel.fit(df_train)
    parallel = time.time() - start
    print(pd.DataFrame(stack_parallel.cv_scores_).mean())
    
    print("Done parallel training...")

    start = time.time()
    parallel_pred = stack_parallel.predict_proba(df_test)[:, 1]
    parallel_predict = time.time() - start

    print("Done parallel predicting...")

    # usual
    start = time.time()
    stack = StackingClassifier(model_dict_list)
    stack.fit(df_train)
    usual = time.time() - start
    
    print(df_test)

    start = time.time()
    usual_pred = stack.predict_proba(df_test)[:, 1]
    usual_predict = time.time() - start
    
    print('parallel:', parallel)
    print('parallel predict:', parallel_predict)
    print('parallel AUC:', metrics.roc_auc_score(df_test.sentiment, parallel_pred))

    print('usual:', usual)
    print('usual_predict:', usual_predict)
    print('usual AUC:', metrics.roc_auc_score(df_test.sentiment, usual_pred))

# todo:
# add postfix to tqdm.


    print(pd.DataFrame(stack.cv_scores_).mean())
    print(pd.DataFrame(stack_parallel.cv_scores_).mean())
