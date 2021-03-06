import time
import numba
import pandas as pd
import numpy as np
import os
from model.ModelBase import  ModelBase
import dill as pickle
import xgboost
import sys

class EnsembleModel(ModelBase):

    d_weight = {
        'lgb': 0.60,
        'xgb': 0.10,
        'gbr': 0.10,
        'rf': 0.10,
        'en': 0.10
    }
    bias_weight = 0.00
    bias = 0.0110
    l_drop_columns = ['logerror', 'parcelid', 'transactiondate', 'index']

    @numba.jit
    def __ApplyEnsemble(self,LgbCol, XgbCol, RfCol, EnCol, GbrCol):
        """"""
        n = len(LgbCol)
        result = np.empty((n), dtype='float32')
        for i in range(n):
            result[i] = (self.d_weight['lgb'] * LgbCol[i] +
                         self.d_weight['xgb'] * XgbCol[i] +
                         self.d_weight['rf'] * RfCol[i] +
                         self.d_weight['en'] * EnCol[i] +
                         self.d_weight['gbr'] * GbrCol[i]) * (1.0 - self.bias_weight) + self.bias * self.bias_weight

        return result

    ## evaluate ensemble model with local MAE
    def EvaluateEnsembleModel(self,InputDir):
        """"""
        start = time.time()

        #lgb_file = '%s/LGB_20170713-12:54:46.pkl' % InputDir
        #lgb_file = '%s/LGB_20170713-20:38:23.pkl' % InputDir ## with nullcount
        #lgb_file = '%s/LGB_20170718-13:23:28.pkl' % InputDir ## with structuretaxvalueratio and landtaxvalueratio
        lgb_file = '%s/LGB_20170726-10:43:51.pkl' % InputDir ## with latitude and longitude optimized
        #xgb_file = '%s/XGB_20170713-13:57:47.pkl' % InputDir
        #xgb_file = '%s/XGB_20170718-14:16:19.pkl' % InputDir
        xgb_file = '%s/XGB_20170719-00:04:35.pkl' % InputDir ## with latitude and longitude optimized
        #rf_file = '%s/RF_20170713-15:00:17.pkl' % InputDir
        #rf_file = '%s/RF_20170718-18:49:01.pkl' % InputDir
        rf_file = '%s/RF_20170719-01:26:15.pkl' % InputDir ## with latitude and longitude optimized
        #en_file = '%s/EN_20170718-18:45:45.pkl' % InputDir
        en_file = '%s/EN_20170719-02:24:08.pkl' % InputDir ## with latitude and longitude optimized
        #gbr_file = '%s/GBR_20170718-13:41:05.pkl' % InputDir
        #gbr_file = '%s/GBR_20170718-13:44:54.pkl' % InputDir
        #gbr_file = '%s/GBR_20170718-18:10:27.pkl' % InputDir
        gbr_file = '%s/GBR_20170719-02:01:49.pkl' % InputDir ## with latitude and longitude optimized

        with open(lgb_file,'rb') as i_file:
            lgb = pickle.load(i_file)
        i_file.close()
        print('Load lgb model done.')
        with open(xgb_file,'rb') as i_file:
            xgb = pickle.load(i_file)
        i_file.close()
        print('Load xgb model done.')
        with open(rf_file,'rb') as i_file:
            rf = pickle.load(i_file)
        i_file.close()
        print('Load rf model done.')
        with open(en_file,'rb') as i_file:
            en = pickle.load(i_file)
        i_file.close()
        print('Load en model done.')
        with open(gbr_file,'rb') as i_file:
            gbr = pickle.load(i_file)
        i_file.close()
        print('Load gbr model done.')
        # with open(lr_file,'rb') as i_file:
        #     lr = pickle.load(i_file)
        # i_file.close()
        # print('Load lr model done.')
        # with open(etr_file,'rb') as i_file:
        #     etr = pickle.load(i_file)
        # i_file.close()
        # print('Load etr model done.')
        # with open(rr_file,'rb') as i_file:
        #     rr = pickle.load(i_file)
        # i_file.close()
        # print('Load rr model done.')

        x_train = self.TrainData.drop(self.l_drop_columns, axis= 1)
        self._l_train_columns = x_train.columns

        pred_ensemble = pd.DataFrame(index=self.ValidData.index)
        pred_ensemble['parcelid'] = self.ValidData['parcelid']

        pred_lgb = pd.DataFrame(index=self.ValidData.index)
        pred_lgb['parcelid'] = self.ValidData['parcelid']

        pred_xgb = pd.DataFrame(index=self.ValidData.index)
        pred_xgb['parcelid'] = self.ValidData['parcelid']

        pred_rf = pd.DataFrame(index=self.ValidData.index)
        pred_rf['parcelid'] = self.ValidData['parcelid']

        pred_en = pd.DataFrame(index=self.ValidData.index)
        pred_en['parcelid'] = self.ValidData['parcelid']

        pred_gbr = pd.DataFrame(index=self.ValidData.index)
        pred_gbr['parcelid'] = self.ValidData['parcelid']
        #
        # pred_lr = pd.DataFrame(index=self.ValidData.index)
        # pred_lr['parcelid'] = self.ValidData['parcelid']
        #
        # pred_etr = pd.DataFrame(index=self.ValidData.index)
        # pred_etr['parcelid'] = self.ValidData['parcelid']
        #
        # pred_rr = pd.DataFrame(index=self.ValidData.index)
        # pred_rr['parcelid'] = self.ValidData['parcelid']

        truth_valid = pd.DataFrame(index=self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        self.ValidData['longitude'] -= -118600000
        self.ValidData['latitude'] -= 34220000

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]
            ## transform latitude and longitude

            ## for common
            x_valid_common = x_valid.drop(['nullcount'],axis= 1)
            ## for lgb
            x_valid_rf = x_valid.copy()
            x_valid_lgb = x_valid.copy()

            x_valid_lgb['structuretaxvalueratio'] = x_valid_lgb['structuretaxvaluedollarcnt'] / x_valid_lgb['taxvaluedollarcnt']
            x_valid_lgb['landtaxvalueratio'] = x_valid_lgb['landtaxvaluedollarcnt'] / x_valid_lgb['taxvaluedollarcnt']
            x_valid_lgb.loc[x_valid_lgb['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
            x_valid_lgb.loc[x_valid_lgb['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1
            #x_valid_lgb['structurelandtaxvaluediff'] = x_valid_lgb['structuretaxvaluedollarcnt'] - x_valid_lgb['landtaxvaluedollarcnt']

            ## for xgb
            x_valid.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_valid.columns]

            #extra_va = pd.read_hdf(path_or_buf='%s/p21/eval_valid_%s.hdf' % (self.InputDir, d), key='valid')
            #x_valid_lr = pd.concat([x_valid_common, extra_va.drop('parcelid', axis= 1)], axis= 1)

            ## add new feature nullcount for lgb, so need to be excluded fo xgb, rf, and en
            dvalid = xgboost.DMatrix(x_valid.drop(['nullcount'],axis= 1))
            ## predict
            pred_lgb_slice = lgb.predict(x_valid_lgb)# * 0.95 + 0.011 * 0.05
            pred_xgb_slice = xgb.predict(dvalid)
            pred_rf_slice = rf.predict(x_valid_rf)# * 0.50 + 0.011 * 0.50
            pred_en_slice = en.predict(x_valid_common)# * 0.80 + 0.011 * 0.20
            pred_gbr_slice = gbr.predict(x_valid_common)# * 0.85 + 0.011 * 0.15
            #pred_lr_slice = lr.predict(x_valid_lr)
            #pred_etr_slice = etr.predict(x_valid_common)
            #pred_rr_slice = rr.predict(x_valid_common)
            ## ensemble
            pred_lgb[d] = pred_lgb_slice
            pred_xgb[d]= pred_xgb_slice
            pred_rf[d] = pred_rf_slice #* (1.0 - bias_weight ) + bias_weight * bias
            pred_en[d] = pred_en_slice #* (1.0 - bias_weight ) + bias_weight * bias
            pred_gbr[d] = pred_gbr_slice #* (1.0 - bias_weight ) + bias_weight * bias
            #pred_lr[d] = pred_lr_slice #* (1.0 - bias_weight ) + bias_weight * bias
            #pred_etr[d] = pred_etr_slice #* (1.0 - bias_weight ) + bias_weight * bias
            #pred_rr[d] = pred_rr_slice #* (1.0 - bias_weight ) + bias_weight * bias
            score = pred_lgb_slice * self.d_weight['lgb'] + \
                    pred_xgb_slice * self.d_weight['xgb'] + \
                    pred_gbr_slice * self.d_weight['gbr'] + \
                    pred_rf_slice * self.d_weight['rf'] + \
                    pred_en_slice * self.d_weight['en']
                    #pred_rr_slice * self.d_weight['rr']
                    #pred_lr_slice * self.d_weight['lr']
                    #pred_etr_slice * self.d_weight['etr']

            pred_ensemble[d] = (1.0 - self.bias_weight) * score + self.bias_weight * self.bias
            df_tmp = self.ValidData[self.ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index,d] = df_tmp['logerror']

        lgb_score = 0.0
        lgb_ae = np.abs(pred_lgb - truth_valid)
        for col in lgb_ae.columns:
            lgb_score += np.sum(lgb_ae[col])
        lgb_score /= len(pred_lgb)

        xgb_score = 0.0
        xgb_ae = np.abs(pred_xgb - truth_valid)
        for col in xgb_ae.columns:
            xgb_score += np.sum(xgb_ae[col])
        xgb_score /= len(pred_xgb)

        rf_score = 0.0
        rf_ae = np.abs(pred_rf - truth_valid)
        for col in rf_ae.columns:
            rf_score += np.sum(rf_ae[col])
        rf_score /= len(pred_rf)

        en_score = 0.0
        en_ae = np.abs(pred_en - truth_valid)
        for col in en_ae.columns:
            en_score += np.sum(en_ae[col])
        en_score /= len(pred_en)

        gbr_score = 0.0
        gbr_ae = np.abs(pred_gbr - truth_valid)
        for col in gbr_ae.columns:
            gbr_score += np.sum(gbr_ae[col])
        gbr_score /= len(pred_gbr)

        # lr_score = 0.0
        # lr_ae = np.abs(pred_lr - truth_valid)
        # for col in lr_ae.columns:
        #     lr_score += np.sum(lr_ae[col])
        # lr_score /= len(pred_lr)

        #
        # etr_score = 0.0
        # etr_ae = np.abs(pred_etr - truth_valid)
        # for col in etr_ae.columns:
        #      etr_score += np.sum(etr_ae[col])
        # etr_score /= len(pred_etr)

        #
        # rr_score = 0.0
        # rr_ae = np.abs(pred_rr - truth_valid)
        # for col in rr_ae.columns:
        #      rr_score += np.sum(rr_ae[col])
        # rr_score /= len(pred_rr)

        ensemble_score = 0.0
        ensemble_ae = np.abs(pred_ensemble - truth_valid)
        for col in ensemble_ae.columns:
            ensemble_score += np.sum(ensemble_ae[col])
        ensemble_score /= len(pred_ensemble)  ##!! divided by number of instances, not the number of 'cells'
        print('=============================')
        print('Local MAE is %.6f(ensemble), %.6f(lgb), %.6f(xgb), %.6f(rf), %.6f(en), %.6f(gbr). ' %
              (ensemble_score, lgb_score, xgb_score, rf_score, en_score, gbr_score))
        print('=============================')

        end = time.time()
        print('time elapsed %ds' % (end - start))

    ## predict for the test data with optimized ensemble model in LOCAL mode
    def SimpleEnsemble(self,InputDir,OutputDir):

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        start = time.time()

        ## ensemble the best ones of lgb and xgb
        #lgb_result = pd.read_csv('%s/lgb_418_biased.csv' % InputDir)
        #lgb_result = pd.read_csv('%s/lgb_418_biased_nullcount.csv' % InputDir)
        lgb_result = pd.read_csv('%s/lgb_lon_lat_optimized.csv' % InputDir)
        xgb_result = pd.read_csv('%s/xgb_lat_lon.csv' % InputDir)  # parameter base_score equals the mean of target
        rf_result = pd.read_csv('%s/rf_lat_lon.csv' % InputDir)
        en_result = pd.read_csv('%s/en_lat_lon.csv' % InputDir)
        gbr_result = pd.read_csv('%s/gbr_lat_lon.csv' % InputDir)

        ensembled_result = pd.DataFrame(index=lgb_result.index)
        ensembled_result['ParcelId'] = lgb_result['ParcelId']

        ensembled_result.set_index('ParcelId', inplace=True)
        lgb_result.set_index('ParcelId', inplace=True)
        xgb_result.set_index('ParcelId', inplace=True)
        rf_result.set_index('ParcelId', inplace=True)
        en_result.set_index('ParcelId', inplace=True)
        gbr_result.set_index('ParcelId', inplace=True)

        ## determined by MAE value of public score

        up_threshold = 0.418

        for col in lgb_result.columns:
            tmp = self.__ApplyEnsemble(lgb_result[col].values,
                                      xgb_result[col].values,
                                      rf_result[col].values,
                                      en_result[col].values,
                                      gbr_result[col].values)
            df_tmp = pd.DataFrame(tmp, index=lgb_result.index, columns=[col])
            ensembled_result = pd.concat([ensembled_result, df_tmp], axis=1)
            print('Enssemble for column %s is done.' % col)

        ensembled_result = ensembled_result.reset_index()
        print('Examples of lgb: ')
        print(lgb_result.head())
        print('Examples of xgb: ')
        print(xgb_result.head())
        print('Examples of ensemble(lgb:xgb:rf:en:gbr=%d:%d:%d:%d:%d), bias weight %.4f' % (int(self.d_weight['lgb']*100),
                                                                               int(self.d_weight['xgb']*100),
                                                                               int(self.d_weight['rf'] * 100),
                                                                               int(self.d_weight['en'] * 100),
                                                                               int(self.d_weight['gbr'] * 100),
                                                                               self.bias_weight)
              )
        print(ensembled_result.head())

        ensemble_sub = '%s/lgb_xgb_rf_en_gbr_%d_%d_%d_%d_%d_%d.csv' % (OutputDir,int(up_threshold * 1000),
                                                          int(self.d_weight['lgb'] * 100),
                                                          int(self.d_weight['xgb'] * 100),
                                                          int(self.d_weight['rf'] * 100),
                                                          int(self.d_weight['en'] * 100),
                                                          int(self.d_weight['gbr'] * 100),
                                                          )
        ensembled_result.to_csv(ensemble_sub, index=False, float_format='%.4f')

        end = time.time()
        print('\nEnsemble of lgb and xgb is done, time consumed %ds' % (end - start))