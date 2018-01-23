from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV   #Perforing grid search

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

def find_depth(x,y):
    param_test1 = {
     'max_depth':list(range(3,10,2)),
     'min_child_weight':list(range(1,6,2)),
     # 'n_estimators':list(range(10,100,10))
    }
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, gamma=0, subsample=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1),
     param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(x,y)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
