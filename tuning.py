from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import pandas as pd

# useful function to find the most important features in the data
# As found on https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
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

# Function to do cross validation using varying parameters
def tune_params(train_x, train_y):
    # uncomment lines and add values to test
    cv_params = {'max_depth': [6,8],
                 #'min_child_weight': [1,3],
                 #'n_estimators': [13,14,15],
                 #'colsample_bytree': [.7,0.8,.9],
                 #'learning_rate': [.001,.01,.1],
                 #'subsample': [.5,.6,.7,.8,.9]
                 }
    ind_params = {'learning_rate': 0.1,
                  'n_estimators': 14,
                  'max_depth': 8,
                  'min_child_weight': 1,
                  'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic'}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                cv_params,
                                scoring = 'accuracy', cv = 100, n_jobs = -1)

    optimized_GBM.fit(train_x, train_y)
    scores = optimized_GBM.grid_scores_
    pdscores = pd.DataFrame(scores)
    #for i in range(3):
    #    print(pdscores['parameters'][i], pdscores['mean_validation_score'][i])
    for item in scores:
        print(item)
