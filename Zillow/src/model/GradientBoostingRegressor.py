from model.ModelBase import ModelBase
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import StratifiedKFold,cross_val_score
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime
import dill as pickle
import math
import gc
import os

class GBR(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index']

    _iter = 120
    _learning_rate = 0.4
    _depth= 5
    _loss = 'lad'

    def train(self):
        """"""
        ## drop noisy columns
        print(self.TrainData.shape)

        N = len(self.TrainData)

        l_drop_select = {'tractcode': .99990,
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
        for sel in l_drop_select:
            Cols = [col for col in self.TrainData.columns if (sel in col)]
            selected = [col for col in Cols if (self.TrainData[col].value_counts().ix[0] > N * l_drop_select[sel])]
            print('%s has %d' % (sel, len(Cols)))
            print('%s was truncted %d' % (sel, len(selected)))
            self.TrainData.drop(selected, axis= 1, inplace= True)

        print('size before truncated outliers is %d ' % len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(TrainData))

        TrainData['longitude'] -= -118600000
        TrainData['latitude'] -= 34220000

        X = TrainData.drop(self._l_drop_cols, axis=1)
        Y = TrainData['logerror']
        self._l_train_columns = X.columns
        FeatCols = list(self._l_train_columns)

        # nfolds = 5
        # BestTreeDepth = 0
        # BestLearningRate = 0
        # MinMeanMetrics = 1000
        # for d in range(4, 5):
        #     for lr in [0.1]:
        #         KFold = StratifiedKFold(self.TrainData['logerror'], nfolds, shuffle=True, random_state=2017)
        #         metrics = np.zeros((len(KFold)), dtype=float)
        #         for fold, (tr, va) in enumerate(KFold, start=0):
        #             print(len(tr),len(va))
        #             TrainData = self.TrainData[self.TrainData.index.isin(tr)].copy()
        #             ValidData = self.TrainData[self.TrainData.index.isin(va)].copy()
        #
        #             gbr = GradientBoostingRegressor(n_estimators= 60,
        #                                         learning_rate= lr,
        #                                         max_depth= d,
        #                                         random_state=2017,
        #                                         subsample=0.8,
        #                                         loss=self._loss,
        #                                         max_features=int(math.sqrt(len(FeatCols))),
        #                                         verbose= False)
        #             gbr.fit(TrainData[FeatCols], TrainData['logerror'])
        #
        #             ValidData['predict'] = gbr.predict(ValidData[FeatCols])
        #             mae = np.sum(np.abs(ValidData['predict'] - ValidData['logerror']))/len(ValidData['predict'])
        #             print('%d : %.6f' % (fold,mae))
        #             metrics[fold] = mae
        #         MeanMetrics = np.mean(metrics)
        #         print("depth %d, lr %.6f, Mean mae %f" % (d, lr, MeanMetrics))
        #         if (MeanMetrics < MinMeanMetrics):
        #             MinMeanMetrics = MeanMetrics
        #             BestTreeDepth = d
        #             BestLearningRate = lr
        #
        # print('%d - %.6f' % (BestTreeDepth, BestLearningRate))
        # sys.exit(1)

        gbr = GradientBoostingRegressor(n_estimators= self._iter,
                                  learning_rate= self._learning_rate,
                                  max_depth = self._depth,
                                  random_state = 2017,
                                  subsample= 0.80,
                                  loss = self._loss,
                                  max_features= int(math.sqrt(len(FeatCols))),
                                  verbose= True)
        self._model = gbr.fit(X,Y)

        ## evaluate on valid data
        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,
                                                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model, 'wb') as o_file:
            pickle.dump(self._model, o_file, -1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],
                                   ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

        return

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
            #x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid)
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
        ## concate train with valid data
        self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

        ## drop noisy columns
        print('train data shape, ', self.TrainData.shape)
        l_drop_cont = []
        with open('%s/drop_selected.dat' % self.OutputDir, 'r') as i_file:
            for line in i_file:
                l_drop_cont.append(line.strip())
        i_file.close()
        self.TrainData.drop(l_drop_cont, axis=1, inplace=True)

        ## retrain with the whole training data
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        self.TrainData['longitude'] -= -118600000
        self.TrainData['latitude'] -= 34220000

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']

        self._l_train_columns = X.columns
        FeatCols = list(self._l_train_columns)
        print('train data shape after cleaned, ', self.TrainData.shape)

        print('feature size %d' % len(self._l_train_columns))

        gbr = GradientBoostingRegressor(n_estimators= self._iter,
                                  learning_rate= self._learning_rate,
                                  max_depth = self._depth,
                                  random_state = 2017,
                                  subsample= 0.80,
                                  loss = self._loss,
                                  max_features= int(math.sqrt(len(FeatCols))),
                                  verbose= True)
        self._model = gbr.fit(X,Y)
        print('training done.')
        del self.TrainData, X, Y
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
        #self.TestData = self.TestData.sample(frac=0.01)

        self._sub = pd.DataFrame(index=self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        self.TestData['longitude'] -= -118600000
        self.TestData['latitude'] -= 34220000
        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N]
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

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,
                                                  datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
