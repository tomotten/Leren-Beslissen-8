from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import pandas as pd

def fit_importance(model, x, y):
    # Fit model using each importance as a threshold
    thresholds = sorted(model.feature_importances_)
    for thresh in thresholds:
    	# select features using threshold
    	selection = SelectFromModel(model, threshold=thresh, prefit=True)
    	select_X_train = selection.transform(x)
    	# train model
    	selection_model = xgb.XGBClassifier()
    	selection_model.fit(select_X_train, y)
    	# eval model
    	select_X_test = selection.transform(x)
    	y_pred = selection_model.predict(select_X_test)
    	predictions = [round(value) for value in y_pred]
    	accuracy = accuracy_score(y, predictions)
    	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

def tune_params(train_x, train_y):
    cv_params = {'max_depth': [5,7,9],
                 'min_child_weight': [1,3,5],
                 'n_estimators': [10,12,14,16]}
                 #'colsample_bytree': [.7,0.8,.9],
                 #'learning_rate': [.005,.01,.015,.02],
                 #'subsample': [.7,.8,.9]}
    ind_params = {'learning_rate': 0.01,
                  'n_estimators': 14,
                  'max_depth': 7,
                  'min_child_weight': 3,
                  'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic'}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                cv_params,
                                scoring = 'accuracy', cv = 50, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation
    optimized_GBM.fit(train_x, train_y)
    scores = optimized_GBM.grid_scores_
    pdscores = pd.DataFrame(scores)
    #for i in range(3):
    #    print(pdscores['parameters'][i], pdscores['mean_validation_score'][i])
    for item in scores:
        print(item)
