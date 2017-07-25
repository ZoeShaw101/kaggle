from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import xgboost
import gc
import math
import time
import dill as pickle
import os
from datetime import datetime
import sys

class XGB(ModelBase):

    _params = {
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': 0.011,
        'silent': 0,
        'npthread': 4,
        'lambda': 0.8,
        'alpha': 0.3995,
        'eta': 0.04,
        'max_depth': 8
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

    _iter = 100

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index']

    def train(self):
        """"""
        ## drop noisy columns
        print(self.TrainData.shape)

        N = len(self.TrainData)

        for sel in self._l_drop_select:
            Cols = [col for col in self.TrainData.columns if (sel in col)]
            selected = [col for col in Cols if (self.TrainData[col].value_counts().ix[0] > N * l_drop_select[sel])]
            print('%s has %d' % (sel, len(Cols)))
            print('%s was truncted %d' % (sel, len(selected)))
            self.TrainData.drop(selected, axis= 1, inplace= True)

        start = time.time()

        print(len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print(len(TrainData))

        TrainData['longitude'] -= -118600000
        TrainData['latitude'] -= 34220000

        x_train = TrainData.drop(self._l_drop_cols, axis= 1)
        y_train = TrainData['logerror'].values.astype(np.float32)
        # Judge if feature selection has been done.
        if(len(self._l_selected_features) == 0):
            print('Full featured ...')
            self._l_train_columns = x_train.columns
        else:
            print('Selected featured ...')
            self._l_train_columns = self._l_selected_features

        self._params['base_score'] = np.mean(y_train)
        dtrain = xgboost.DMatrix(x_train, y_train)

        ## parameter tuning with CV
        # BestParams = {'eta':0.0,'max_depth':0}
        # BestMAE = 1.0
        # l_eta = [0.03*math.pow(2.0,i) for i in range(1)]
        # l_max_depth = [(6 + v) for v in range(5)]
        # for eta in l_eta:
        #     for depth in l_max_depth:
        #         params['eta'] = eta
        #         params['max_depth'] = depth
        #         print( "Running XGBoost CV ..." )
        #         cv_result = xgboost.cv(params,
        #                   dtrain,
        #                   nfold=5,
        #                   num_boost_round=100,
        #                   early_stopping_rounds=50,
        #                   verbose_eval=10,
        #                   show_stdv=True
        #                  )
        #         if(cv_result.get('test-mae')[-1] < BestMAE):
        #             BestMAE = cv_result.get('test-mae')[-1]
        #             BestParams['eta'] = eta
        #             BestParams['max_depth'] = depth
        # print(BestParams)

        ## train model
        print("\nTraining XGBoost ...")
        self._model = xgboost.train(self._params, dtrain, num_boost_round= self._iter)

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model,'wb') as o_file:
            pickle.dump(self._model,o_file,-1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    def evaluate(self):
        """"""
        ValidData = self.ValidData

        ValidData['longitude'] -= -118600000
        ValidData['latitude'] -= 34220000

        pred_valid = pd.DataFrame(index= ValidData.index)
        pred_valid['parcelid'] = ValidData['parcelid']

        truth_valid = pd.DataFrame(index= ValidData.index)
        truth_valid['parcelid'] = ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = ValidData[l_valid_columns]
            x_valid.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_valid.columns]
            dvalid = xgboost.DMatrix(x_valid)
            pred_valid[d] = self._model.predict(dvalid)
            df_tmp = ValidData[ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index, d] = df_tmp['logerror']

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

    def submit(self):
        """"""
        start = time.time()
        ## concate train with valid
        print('train data shape before concated, ', self.TrainData.shape)
        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        ## drop noisy columns
        print('train data shape after concated, ', self.TrainData.shape)
        l_drop_cont = []
        with open('%s/drop_selected.dat' % self.OutputDir, 'r') as i_file:
            for line in i_file:
                l_drop_cont.append(line.strip())
        i_file.close()
        self.TrainData.drop(l_drop_cont, axis= 1, inplace= True)

        ## retrain model
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        self.TrainData['longitude'] -= -118600000
        self.TrainData['latitude'] -= 34220000

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror'].values.astype(np.float32)
        self._params['base_score'] = np.mean(Y)

        self._l_train_columns = X.columns
        print('train data shape after cleaned, ', self.TrainData.shape)

        print('feature size %d' % len(self._l_train_columns))

        dtrain = xgboost.DMatrix(X,Y)

        print('\n Retraining XGBoost ...')
        self._model = xgboost.train(self._params, dtrain, num_boost_round= self._iter)
        print('\n Retraining done.')

        del X,Y,dtrain,self.TrainData
        gc.collect()

        ## for test
        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
        #self.TestData = self.TestData.sample(frac = 0.01)

        self._sub = pd.DataFrame(index=self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']


        self.TestData['longitude'] -= -118600000
        self.TestData['latitude'] -= 34220000
        N = 200000
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print("Start prediction ...")
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                              self._l_train_columns]
            x_test = self.TestData[l_test_columns]
            x_test.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_test.columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N]
                dtest = xgboost.DMatrix(x_test_block)
                ret = self._model.predict(dtest)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done, time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()

        end = time.time()
        print('Prediction is done, time elapsed %ds' % (end - start))

        if (os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
        print('Submit is done.')
