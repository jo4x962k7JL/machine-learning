#
import warnings, time, gc
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
start = time.time()
RANDOM = 47

data = pd.read_csv('./application_train_final.csv')
test = pd.read_csv('./application_test_final.csv')

y_train = data['TARGET']
X_train = data.drop(columns = ['SK_ID_CURR', 'TARGET'])
submission = test[['SK_ID_CURR']].copy()
X_test = test.drop(columns = ['SK_ID_CURR'])

del data, test
gc.collect()

print('X_train size: {}\nX_test size: {}'.format(X_train.shape, X_test.shape))

# train and predict
oof_pred = np.zeros(X_train.shape[0])
y_pred = np.zeros(X_test.shape[0])
folds = KFold(n_splits= 5, shuffle=True, random_state=RANDOM)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    clf = LGBMClassifier(nthread=-1)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 50)
    oof_pred[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_pred += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_pred[valid_idx])))
print('Full AUC score %.6f' % roc_auc_score(y_train, oof_pred))

# submit
submission['TARGET'] = y_pred
print(submission.head())
submission.to_csv('csv_sim3_fe1_cv.csv', index = False)
print('Run time: {:.2f}mins'.format((time.time() - start)/60))
