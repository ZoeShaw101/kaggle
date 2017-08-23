import time
from model.LightGBM import LGB

class SingleModel:

    @staticmethod
    def __LaunchTraining(task, kfold, InputDir, OutputDir):

        d_model = {
            'lgb': LGB
                   }

        ## training for single model with cv on data of 1-6
        print('Training for %s begins ...' % task)
        start = time.time()

        l_mae = []
        for fold in range(kfold):
            FoldInputDir= '%s/train/%s' % (InputDir, fold)
            model = d_model[task]([7 - (kfold - fold) + 1], FoldInputDir,OutputDir)
            TrainMAE = model.train()
            l_mae.append(TrainMAE)

        end = time.time()
        print('Training for %s done, time elapsed %ds' % (task,(end - start)))

        ## test single model on untouched data of July
        TestInputDir = '%s/test' % InputDir
        model = d_model[task]([8], TestInputDir, OutputDir)
        TestMAE = model.train()

        print('\n==== Summary for single model %s ====\n' % task)
        CVMAE = 0
        for fold in range(len(l_mae)):
            mae = l_mae[fold]
            print('---- fold %d training on data 1-%d, mae %.6f on test data %d' % (fold, 7 - (kfold - fold), mae, 7 - (kfold - fold) + 1))
            CVMAE += mae
        CVMAE /= len(l_mae)

        print('\n---- local cv mae : %.6f' % CVMAE)
        print('---- local mae %.6f on test data %d ' % (TestMAE, 8))

        return

    @classmethod
    def run(cls, strategies, kfold, InputDir, OutputDir):

        start = time.time()
        for s in strategies:
            cls.__LaunchTraining(s, kfold, InputDir, OutputDir)
        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))