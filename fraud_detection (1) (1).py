import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv('/content/train_fraud.csv')
test_data = pd.read_csv('/content/test_fraud.csv')
print("Train Data Columns: ", train_data.columns)
print("Test Data Columns: ", test_data.columns)
print(train_data.info())
print(test_data.info())
print(train_data.head())
print(test_data.head())
print(train_data.tail())
print(test_data.tail())
print(train_data.shape)
print(test_data.shape)
# Drop non-numeric columns that can't be used directly for modeling
columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long',
                   'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

X_train = train_data.drop(columns_to_drop + ['is_fraud'], axis=1)
y_train = train_data['is_fraud']

X_test = test_data.drop(columns_to_drop + ['is_fraud'], axis=1)
y_test = test_data['is_fraud']
# Handle missing values in features
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Handle missing values in target variable
y_train = y_train.fillna(y_train.mode()[0])
y_test = y_test.fillna(y_test.mode()[0])

# Scale the 'amt' feature
scaler = StandardScaler()
X_train['amt'] = scaler.fit_transform(X_train[['amt']])
X_test['amt'] = scaler.transform(X_test[['amt']])
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg, target_names=['fraudulent', 'legitimate']))
print('ROC AUC:', roc_auc_score(y_test, y_pred_log_reg))
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("Decision Tree:")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree, target_names=['fraudulent', 'legitimate']))
print('ROC AUC:', roc_auc_score(y_test, y_pred_tree))
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print("Random Forest:")
print(confusion_matrix(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest, target_names=['fraudulent', 'legitimate']))
print('ROC AUC:', roc_auc_score(y_test, y_pred_forest))
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Tuned Random Forest:")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print('Best ROC AUC:', roc_auc_score(y_test, y_pred_best))
