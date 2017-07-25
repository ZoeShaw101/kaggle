from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import lightgbm
import gc
import os
import time
import math
from datetime import datetime
import numba
import dill as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold,cross_val_score
from sklearn.grid_search import GridSearchCV
import sys
from sklearn.decomposition import TruncatedSVD

class LGB(ModelBase):
    """"""
    _params = {
        'max_bin': 8,
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'lambda_l1': 0.4,
        'sub_feature': 0.70,
        'bagging_fraction':  0.80,
        'num_leaves': 128,
        'min_data':  150,
        'min_hessian':  0.01,
        'learning_rate': 0.02,
        'bagging_freq': 15
    }

    _l_drop_select = {'tractcode': .99990,
                     'regionidzip': .9995,
                     'regionidneighborhood': .9995,
                     'regionidcity': .9995,
                     'propertylandusetypeid': .999,
                     'heatingorsystemtypeid': .999,
                     'buildingqualitytypeid': .999,
                     'architecturalstyletypeid': .999,
                     'airconditioningtypeid': .999,
                     'blockcode': .999
                     }

    _iter = 600

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index']

    ## rewritten method
    def train(self):
        """"""
        start = time.time()

        ## drop noisy columns
        l_drop_cont = []
        print(self.TrainData.shape)
        N = len(self.TrainData)
        for sel in self._l_drop_select:
            Cols = [col for col in self.TrainData.columns if (sel in col)]
            selected = [col for col in Cols if (self.TrainData[col].value_counts().ix[0] > N * self._l_drop_select[sel])]
            print('%s has %d' % (sel, len(Cols)))
            print('%s was truncted %d' % (sel, len(selected)))
            self.TrainData.drop(selected, axis= 1, inplace= True)
            l_drop_cont.extend(selected)

        with open('%s/drop_selected.dat' % self.OutputDir, 'w') as o_file:
            for col in l_drop_cont:
                o_file.write('%s\n' % col)
        o_file.close()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(TrainData))

        #with open('/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data/CensusFeat.pkl', 'rb') as i_file:
        #    df_cf = pickle.load(i_file)
        #i_file.close()
        #df_cf['tractcode'] = df_cf['tractcode'].astype(int)
        #df_cf['blockcode'] = df_cf['blockcode'].astype(int)
        #TrainData = TrainData.merge(df_cf.drop(['fipscode', 'blockcode'], axis= 1), how='left', on='parcelid')

        #TrainData['bathroomratio'] = TrainData['bathroomcnt'] / TrainData['calculatedbathnbr']
        #TrainData.loc[TrainData['bathroomratio'] < 0, 'bathroomratio'] = -1

        # TrainData['calculatedfinishedsquarefeatratio'] = TrainData['calculatedfinishedsquarefeet'] / TrainData['lotsizesquarefeet']
        # TrainData.loc[TrainData['calculatedfinishedsquarefeatratio'] > 1, 'calculatedfinishedsquarefeatratio'] = 1
        # TrainData.loc[TrainData['calculatedfinishedsquarefeatratio'] < 0, 'calculatedfinishedsquarefeatratio'] = -1

        # TrainData['finishedsquarefeatratio'] = TrainData['finishedfloor1squarefeet'] / TrainData['lotsizesquarefeet']
        # TrainData.loc[TrainData['finishedsquarefeatratio'] > 1, 'finishedsquarefeatratio'] = 1
        # TrainData.loc[TrainData['finishedsquarefeatratio'] < 0, 'finishedsquarefeatratio'] = -1

        TrainData['longitude'] -= -118600000
        TrainData['latitude'] -= 34220000
        TrainData.drop('roomcnt', axis= 1, inplace= True)
        TrainData.drop('finishedsquarefeet12', axis= 1, inplace= True)

        #TrainData['yearbuilt'] -= 1960
        #TrainData['unitcnt'] -= 2
        #TrainData['finishedsquarefeet15'] -= 6000
        #TrainData['finishedsquarefeet15'] -= 5000
        #TrainData['longitude2'] = TrainData['longitude'] - (-118100000)

        #TrainData.drop('latitude', axis= 1, inplace= True)
        #TrainData.drop('longitude', axis= 1, inplace= True)

        #TrainData['quarter'] = TrainData['transactiondate'].dt.quarter

        #TrainData.loc[TrainData['taxamount'] > 1, 'taxamount'] = np.log(TrainData[TrainData['taxamount'] > 1]['taxamount'])
        #TrainData.loc[TrainData['taxamount'] <= 1, 'taxamount'] = 0

        # TrainData.loc[TrainData['roomcnt'] > 0, 'bedroomratio'] = (1.0 * TrainData['bedroomcnt']) / TrainData['roomcnt']
        # TrainData.loc[TrainData['roomcnt'] <= 0, 'bedroomratio'] = -1
        # TrainData.loc[TrainData['bedroomratio'] <= 0, 'bedroomratio'] = -1

        #TrainData['finishedfloor1ratio'] = TrainData['finishedfloor1squarefeet'] / TrainData['finishedsquarefeet50']
        #TrainData.loc[TrainData['finishedfloor1ratio'] < 0, 'finishedfloor1ratio'] = -1

        #TrainData['finishedlivingratio'] = TrainData['finishedsquarefeet12'] / TrainData['calculatedfinishedsquarefeet']
        #TrainData.loc[TrainData['finishedlivingratio'] < 0, 'finishedlivingratio'] = -1

        #TrainData['bathbeddiff'] = TrainData['calculatedbathnbr'] - TrainData['bedroomcnt']
        #TrainData.loc[TrainData['lotsizesquarefeet'] > 1024000, 'lotsizesquarefeet'] = 1024000
        #TrainData.drop(['longitude','latitude'], axis= 1, inplace= True)
        #TrainData['latitude1'] = (TrainData['latitude'] / 10).astype(int)
        #TrainData['latitude2'] = TrainData['latitude'].astype(int) % 10
        #TrainData = TrainData.drop(['latitude'], axis= 1)

        TrainData['structuretaxvalueratio'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1

        TrainData['landtaxvalueratio'] = TrainData['landtaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        #TrainData.loc[TrainData['taxvaluedollarcnt'] >= 1, 'taxvaluedollarcnt'] = np.log(TrainData[TrainData['taxvaluedollarcnt'] >= 1]['taxvaluedollarcnt'])
        #TrainData.loc[TrainData['taxvaluedollarcnt'] < 1, 'taxvaluedollarcnt'] = -1
        #TrainData['structurelandtaxvaluediff'] = TrainData['structuretaxvaluedollarcnt'] - TrainData['landtaxvaluedollarcnt']
        #TrainData['theothertaxvalueratio'] = 1.0 - TrainData['structuretaxvalueratio'] - TrainData['landtaxvalueratio']

        #TrainData['propertytaxratio'] = TrainData['taxamount'] / TrainData['taxvaluedollarcnt']
        #TrainData.loc[TrainData['propertytaxratio'] < 0, 'propertytaxratio'] = -1
        #TrainData.loc[TrainData['propertytaxratio'] > 1, 'propertytaxratio'] = 1

        TrainData['taxvaluedollarcnt'] -= 750000

        #fips_cols = [col for col in TrainData.columns if(col.startswith('fips'))]
        #TrainData.drop(fips_cols, axis= 1, inplace= True)
        # census_drop_cols = [col for col in TrainData.columns if(col.startswith('rawcensustractandblock'))]
        # TrainData = TrainData.drop(census_drop_cols, axis= 1)

        X = TrainData.drop(self._l_drop_cols,axis= 1)
        Y = TrainData['logerror']

        ## features not been selected yet
        if(len(self._l_selected_features) == 0):
            self._l_train_columns = X.columns
        else:
            self._l_train_columns = self._l_selected_features

        print('feature number %d' % len(self._l_train_columns))

        ## feature transform with svd
        # self._trans_model = TruncatedSVD(n_components= 100, n_iter= 10, random_state= 2017)
        # X = X.values.astype(np.float32, copy=False)
        # X = self._trans_model.fit_transform(X)
        # print('feature is transformed.')


        d_cv = lightgbm.Dataset(X,label=Y)

        ## one-hold mode for parameter tuning
        # msk = np.random.rand(len(self.TrainData)) < 0.9
        # train = self.TrainData[msk]
        # valid = self.TrainData[~msk]
        # x_train = train.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_train = train['logerror']
        # self._l_train_columns = x_train.columns

        # x_valid = valid.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_valid = valid['logerror']
        #
        # x_train = x_train.values.astype(np.float32, copy=False)
        # x_valid = x_valid.values.astype(np.float32, copy=False)
        #
        # d_train = lightgbm.Dataset(x_train,label=y_train)
        # d_valid = lightgbm.Dataset(x_valid,label=y_valid)
        # params['learning_rate'] = 0.026
        # params['bagging_freq'] = 20
        # self._model = lightgbm.train(params,d_train,100,verbose_eval= True,valid_sets=[d_valid])

        ## cv mode for parameter tuning
        # l_learning_rate = [0.014 + 0.002*i for i in range(5)]
        # l_bagging_freq = [10 + i*10 for i in range(5)]
        #
        # BestParams = {'learning_rate':0.0,'bagging_freq':0}
        # BestMAE = 1.0
        # for lr in l_learning_rate:
        #     for bf in l_bagging_freq:
        #         params['learning_rate'] = lr
        #         params['bagging_freq'] = bf
        #
        #         self._model = lightgbm.cv(params, d_cv, 100, nfold=5,verbose_eval= True)
        #         if(self._model.get('l1-mean')[-1] < BestMAE):
        #             BestMAE = self._model.get('l1-mean')[-1]
        #             BestParams['learning_rate'] = lr
        #             BestParams['bagging_freq'] = bf
        # print(BestParams)
        # params['learning_rate'] = BestParams['learning_rate']
        # params['bagging_freq'] = BestParams['bagging_freq']

        #NewCols = ['col%d' % n for n,col in enumerate(list(X.columns))]
        #X.columns = NewCols
        self._model = lightgbm.train(self._params,
                                     d_cv,
                                     self._iter,
                                     verbose_eval= True)


        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model,'wb') as o_file:
            pickle.dump(self._model,o_file,-1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    ## evaluate on valid data
    def evaluate(self):
        """"""
        ValidData = self.ValidData

        #with open('/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data/CensusFeat.pkl', 'rb') as i_file:
        #    df_cf = pickle.load(i_file)
        #i_file.close()
        #df_cf['tractcode'] = df_cf['tractcode'].astype(int)
        #df_cf['blockcode'] = df_cf['blockcode'].astype(int)
        #ValidData = ValidData.merge(df_cf.drop(['fipscode', 'blockcode'], axis= 1), how='left', on='parcelid')

        # with open('/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data/CountFeat.pkl', 'rb') as i_file:
        #     df_cf = pickle.load(i_file)
        # i_file.close()
        # ValidData = ValidData.merge(df_cf, how='left', on='parcelid')

        #ValidData['bathroomratio'] = ValidData['bathroomcnt'] / ValidData['calculatedbathnbr']
        #ValidData.loc[ValidData['bathroomratio'] < 0, 'bathroomratio'] = -1

        # ValidData['calculatedfinishedsquarefeatratio'] = ValidData['calculatedfinishedsquarefeet'] / ValidData['lotsizesquarefeet']
        # ValidData.loc[ValidData['calculatedfinishedsquarefeatratio'] > 1, 'calculatedfinishedsquarefeatratio'] = 1
        # ValidData.loc[ValidData['calculatedfinishedsquarefeatratio'] < 0, 'calculatedfinishedsquarefeatratio'] = -1

        # ValidData['finishedsquarefeatratio'] = ValidData['finishedfloor1squarefeet'] / ValidData['lotsizesquarefeet']
        # ValidData.loc[ValidData['finishedsquarefeatratio'] > 1, 'finishedsquarefeatratio'] = 1
        # ValidData.loc[ValidData['finishedsquarefeatratio'] < 0, 'finishedsquarefeatratio'] = -1

        ValidData['longitude'] = ValidData['longitude'] - (-118600000)
        ValidData['latitude'] = ValidData['latitude'] - 34220000
        ValidData.drop('roomcnt', axis= 1, inplace= True)
        ValidData.drop('finishedsquarefeet12', axis= 1, inplace= True)

        #ValidData['yearbuilt'] -= 1960
        #ValidData['unitcnt'] -= 2
        #ValidData['finishedsquarefeet15'] -= 6000
        #ValidData['finishedsquarefeet15'] -= 5000
        #ValidData['longitude2'] = ValidData['longitude'] - (-118100000)

        #ValidData.drop('longitude', axis= 1, inplace= True)
        #ValidData.drop('latitude', axis= 1, inplace= True)

        #ValidData['quarter'] = ValidData['transactiondate'].dt.quarter
        #ValidData.loc[ValidData['taxamount'] > 1, 'taxamount'] = np.log(ValidData[ValidData['taxamount'] > 1]['taxamount'])
        #ValidData.loc[ValidData['taxamount'] <= 1, 'taxamount'] = 0

        # ValidData.loc[ValidData['roomcnt'] > 0, 'bedroomratio'] = (1.0 * ValidData['bedroomcnt']) / ValidData['roomcnt']
        # ValidData.loc[ValidData['roomcnt'] <= 0, 'bedroomratio'] = -1
        # ValidData.loc[ValidData['bedroomratio'] <= 0, 'bedroomratio'] = -1

        #ValidData['finishedfloor1ratio'] = ValidData['finishedfloor1squarefeet'] / ValidData['finishedsquarefeet50']
        #ValidData.loc[ValidData['finishedfloor1ratio'] < 0, 'finishedfloor1ratio'] = -1
        #ValidData['finishedlivingratio'] = ValidData['finishedsquarefeet12'] / ValidData['calculatedfinishedsquarefeet']
        #ValidData.loc[ValidData['finishedlivingratio'] < 0, 'finishedlivingratio'] = -1
        #ValidData['finishedfirstfloorratio'] = ValidData['finishedsquarefeet6'] / ValidData['calculatedfinishedsquarefeet']
        #ValidData['bathbeddiff'] = ValidData['calculatedbathnbr'] - ValidData['bedroomcnt']
        #ValidData.loc[ValidData['lotsizesquarefeet'] > 1024000, 'lotsizesquarefeet'] = 1024000
        #ValidData.drop(['longitude','latitude'], axis= 1, inplace= True)
        #ValidData['latitude1'] = (ValidData['latitude'] / 10).astype(int)
        #ValidData['latitude2'] = ValidData['latitude'].astype(int) % 10
        #ValidData = ValidData.drop(['latitude'], axis= 1)
        #ValidData['longitude1'] = (ValidData['longitude'] / 10).astype(int)
        #ValidData['longitude2'] = ValidData['longitude'].astype(int) % 1000

        ValidData['structuretaxvalueratio'] = ValidData['structuretaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        ValidData.loc[ValidData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1

        ValidData['landtaxvalueratio'] = ValidData['landtaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        ValidData.loc[ValidData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1


        #ValidData.loc[ValidData['taxvaluedollarcnt'] >= 1, 'taxvaluedollarcnt'] = np.log(ValidData[ValidData['taxvaluedollarcnt'] >= 1]['taxvaluedollarcnt'])
        #ValidData.loc[ValidData['taxvaluedollarcnt'] < 1, 'taxvaluedollarcnt'] = -1
        #ValidData['structurelandtaxvaluediff'] = ValidData['structuretaxvaluedollarcnt'] - ValidData['landtaxvaluedollarcnt']
        #ValidData['theothertaxvalueratio'] = 1.0 - ValidData['structuretaxvalueratio'] - ValidData['landtaxvalueratio']

        #ValidData['propertytaxratio'] = ValidData['taxamount'] / ValidData['taxvaluedollarcnt']
        #ValidData.loc[ValidData['propertytaxratio'] < 0, 'propertytaxratio'] = -1
        #ValidData.loc[ValidData['propertytaxratio'] > 1, 'propertytaxratio'] = 1

        ValidData['taxvaluedollarcnt'] -= 750000

        pred_valid = pd.DataFrame(index = ValidData.index)
        pred_valid['parcelid'] = ValidData['parcelid']

        truth_valid = pd.DataFrame(index = ValidData.index)
        truth_valid['parcelid'] = ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = ValidData[l_valid_columns]
            pred_valid[d] = self._model.predict(x_valid)
            df_tmp = ValidData[ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index,d] = df_tmp['logerror']

        score = 0.0
        ae = np.abs(pred_valid - truth_valid)
        for col in ae.columns:
            score += np.sum(ae[col])
        score /= len(pred_valid)  ##!! divided by number of instances, not the number of 'cells'
        print('============================= ')
        print('Local MAE is %.6f' % score)
        print('=============================')

        end = time.time()

        del self.ValidData
        gc.collect()

        print('time elapsed %ds' % (end - start))

    @numba.jit
    def __ApplyAE(self,PredColumn,TruthColumn):

        n = len(PredColumn)
        result = np.empty(n,dtype= 'float32')
        for i in range(n):
            v = TruthColumn[i]
            if(math.isnan(v) == False):
                result[i] = np.abs(v - PredColumn[i])

        return result

    def __ComputeMAE(self,df_pred,df_truth):
        """"""
        mae = pd.DataFrame(index = df_pred)
        for col in df_pred.columns:
            if(col == 'parcelid'):
                continue
            ret = self.__ApplyAE(df_pred[col],df_truth[col])
            mae[col] = pd.Series(ret,index= df_pred.index)

    ## predict on test data
    def submit(self):

        ## concate train with valid data
        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        ## drop noisy columns
        print('train data shape, ', self.TrainData.shape)
        l_drop_cont = []
        with open('%s/drop_selected.dat' % self.OutputDir, 'r') as i_file:
            for line in i_file:
                l_drop_cont.append(line.strip())
        i_file.close()
        self.TrainData.drop(l_drop_cont, axis= 1, inplace= True)
        # N = len(self.TrainData)
        # for sel in self._l_drop_select:
        #     Cols = [col for col in self.TrainData.columns if (sel in col)]
        #     selected = [col for col in Cols if (self.TrainData[col].value_counts().ix[0] > N * self._l_drop_select[sel])]
        #     print('%s has %d' % (sel, len(Cols)))
        #     print('%s was truncted %d' % (sel, len(selected)))
        #     self.TrainData.drop(selected, axis= 1, inplace= True)

        ## retrain with the whole training data
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        self.TrainData['longitude'] -= -118600000
        self.TrainData['latitude'] -= 34220000

        self.TrainData.drop('roomcnt', axis= 1, inplace= True)
        self.TrainData.drop('finishedsquarefeet12', axis= 1, inplace= True)

        self.TrainData['structuretaxvalueratio'] = self.TrainData['structuretaxvaluedollarcnt'] / self.TrainData['taxvaluedollarcnt']
        self.TrainData['landtaxvalueratio'] = self.TrainData['landtaxvaluedollarcnt'] / self.TrainData['taxvaluedollarcnt']
        self.TrainData.loc[self.TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        self.TrainData.loc[self.TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        self.TrainData['taxvaluedollarcnt'] -= 750000

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']

        self._l_train_columns = X.columns
        print('train data shape after cleaned, ', self.TrainData.shape)

        print('feature size %d' % len(self._l_train_columns))

        X = X.values.astype(np.float32, copy=False)
        d_train = lightgbm.Dataset(X, label=Y)

        self._model = lightgbm.train(self._params, d_train, self._iter, verbose_eval=True)
        print('training done.')

        del self.TrainData, X, Y, d_train
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir,'test')
        #self.TestData = self.TestData.sample(frac = 0.01)

        self._sub = pd.DataFrame(index = self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        self.TestData['longitude'] -= -118600000
        self.TestData['latitude'] -= 34220000

        self.TestData.drop('roomcnt', axis= 1, inplace= True)
        self.TestData.drop('finishedsquarefeet12', axis= 1, inplace= True)

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)
            self.TestData['structuretaxvalueratio'] = self.TestData['structuretaxvaluedollarcnt'] / self.TestData['taxvaluedollarcnt']
            self.TestData['landtaxvalueratio'] = self.TestData['landtaxvaluedollarcnt'] / self.TestData['taxvaluedollarcnt']
            self.TestData.loc[self.TestData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
            self.TestData.loc[self.TestData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

            self.TestData['taxvaluedollarcnt'] -= 750000

            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
                self._model.reset_parameter({"num_threads": 4})
                ret = self._model.predict(x_test_block)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done. time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()

        end = time.time()
        print('Prediction is done. time elapsed %ds' % (end - start))

        if (os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
