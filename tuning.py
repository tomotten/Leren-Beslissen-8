from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import accuracy_score

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

def find_depth():
    
