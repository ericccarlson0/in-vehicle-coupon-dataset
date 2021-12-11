# %% IMPORTS

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb

from typing import List, Tuple, Union
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

# %% LOAD DATA. PREPROCESS DATA.

DATASET_DIR: str = "/Users/ericcarlson/Desktop/Duke/Semester 6/COMPSCI_671/Final Project/Data"

rename_columns_: dict = {
    'Driving_to': 'DrivingTo',
    'Passanger': 'Passenger',
    'Coupon_validity': 'CouponValidity',
    'Maritalstatus': 'MaritalStatus',
    'Restaurantlessthan20': 'Restaurant0to20',
    'Direction_same': 'DirectionSame',
}
drop_columns_: List = ['id']
coupon_types_: List = ['Bar', 'Coffeehouse', 'Carryaway', 'Restaurant0to20', 'Restaurant20to50']
categorical_columns_: List = ['Coupon', 'DrivingTo', 'Passenger', 'CouponValidity', 'MaritalStatus',
                              'Weather', 'Time', 'DirectionSame', 'Distance', 'Bar', 'Coffeehouse',
                              'Carryaway', 'Restaurant0to20', 'Restaurant20to50', 'Age', 'Education',
                              'Occupation', 'Income', 'Gender', 'Children']
boolean_columns_: List = ['Decision', 'Children', 'DirectionSame']

df_: pd.DataFrame
df_ = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
test_df_ = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

print(df_['Temperature'].value_counts())

def preprocess(df: pd.DataFrame):
    df.rename(columns=rename_columns_, inplace=True)
    df.drop(columns=drop_columns_, inplace=True)

    df['Bar'].fillna(value=-1, inplace=True)
    df['Coffeehouse'].fillna(value=-1, inplace=True)
    df['Carryaway'].fillna(value=-1, inplace=True)
    df['Restaurant0to20'].fillna(value=-1, inplace=True)
    df['Restaurant20to50'].fillna(value=-1, inplace=True)

    df['Coupon'].replace('Carry out & Take away', 'Carryaway', inplace=True)
    df['Coupon'].replace('Coffee House', 'Coffeehouse', inplace=True)
    df['Coupon'].replace('Restaurant(<20)', 'Restaurant0to20', inplace=True)
    df['Coupon'].replace('Restaurant(20-50)', 'Restaurant20to50', inplace=True)

    # print("PRE", df_.memory_usage(deep=True).sum())
    for colname in categorical_columns_:
        if colname not in df.columns:
            continue
        df[colname] = pd.Categorical(df[colname])
    for colname in boolean_columns_:
        if colname not in df.columns:
            continue
        df[colname] = df[colname].astype(bool)
    # print("POST:", df_.memory_usage(deep=True).sum())

    df['Distance'].cat.reorder_categories([1, 2, 3], ordered=True, inplace=True)
    df['Age'].cat.reorder_categories(['below21', '21', '26', '31', '36', '41', '46', '50plus'],
                                     ordered=True, inplace=True)
    # print(df_['Distance'].cat.as_ordered())
    # print(df_['Age'].cat.as_ordered())

preprocess(df_)
preprocess(test_df_)

# %% SEPARATE TRAINING AND TEST DATA. SEGMENT DATA BASED ON COUPON TYPE

X_: pd.DataFrame
y_: pd.Series

X_ = df_.drop(columns=['Decision'])
y_ = df_['Decision']

X_train_: pd.DataFrame
X_test_: pd.DataFrame
y_train_: pd.Series
y_test_: pd.Series

X_train_, X_test_, y_train_, y_test_ = train_test_split(pd.get_dummies(X_),
                                                        y_, test_size=.15)

max_feature_count_: int = 0
for colname_ in X_.columns:
    print(f"{colname_}: {len(X_[colname_].value_counts())}")

# %% RANDOM FORESTS IN SKLEARN

def avg_node_counts(random_forest) -> float:
    node_count = 0
    tree_count = 0

    for tree in random_forest.estimators_:
        node_count += tree.tree_.node_count
        tree_count += 1
        if tree_count >= 2**10:
            break

    return node_count / tree_count

def avg_depths(random_forest) -> float:
    depth_count = 0
    tree_count = 0

    for tree in random_forest.estimators_:
        depth_count += tree.tree_.max_depth
        tree_count += 1
        if tree_count >= 2**10:
            break

    return depth_count / tree_count

def train_with_ccp_alphas(X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series],
                          ccp_alphas, **kwargs) -> dict:
    rf_dict = {}
    for alpha in ccp_alphas:
        rf = RandomForestClassifier(ccp_alpha=alpha, **kwargs)
        t0 = time()
        print(f"ALPHA={alpha:.5f}...")
        rf.fit(X_train, y_train)
        print(f"{time() - t0: .3f} secs.")
        rf_dict[alpha] = rf

    return rf_dict

def plot_rf_accuracies(ccp_alphas: List[float], rf_dict: dict, X_train, y_train, X_test, y_test):
    train_scores: List[float] = []
    test_scores: List[float] = []
    for alpha in ccp_alphas:
        rf = rf_dict[alpha]
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))

    fig, ax = plt.subplots()
    ax.set_xlabel("ALPHA")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.0, 1.0])
    ax.plot(ccp_alphas, train_scores, color='k', label='train', drawstyle='steps-post')
    ax.plot(ccp_alphas, test_scores, color='m', label='test', drawstyle='steps-post')
    ax.legend()
    plt.show()

    print(f"Max test score: {max(test_scores):.4f}")

def plot_rf_node_counts(ccp_alphas: List[float], rf_dict: dict):
    node_counts: List[float] = []
    for alpha in ccp_alphas:
        rf = rf_dict[alpha]
        node_counts.append(avg_node_counts(rf))

    fig, ax = plt.subplots()
    ax.set_xlabel("ALPHA")
    ax.set_ylabel("Total Number Nodes")
    ax.set_ylim(0, max(node_counts)*1.1)
    ax.plot(ccp_alphas, node_counts, color='k', drawstyle='steps-post')
    plt.show()

def plot_rf_depth_counts(ccp_alphas: List[float], rf_dict: dict):
    depth_counts: List[float] = []
    for alpha in ccp_alphas:
        rf = rf_dict[alpha]
        depth_counts.append(avg_depths(rf))

    fig, ax = plt.subplots()
    ax.set_xlabel("ALPHA")
    ax.set_ylabel("Tree Depth")
    ax.set_ylim(0, max(depth_counts)*1.1)
    ax.plot(ccp_alphas, depth_counts, color='k', drawstyle='steps-post')
    plt.show()

ccp_alphas_: List
rf_dict_: dict
# TODO: UNCOMMENT
# ccp_alphas_ = np.arange(1, 25) * 1e-4 # np.arange(1, 10) * 1e-3
# rf_dict_ = train_with_ccp_alphas(X_train_, y_train_, ccp_alphas_)
# plot_rf_accuracies(ccp_alphas_, rf_dict_, X_train_, y_train_, X_test_, y_test_)
# plot_rf_node_counts(ccp_alphas_, rf_dict_)
# plot_rf_depth_counts(ccp_alphas_, rf_dict_)

# %% CROSS VALIDATION GRID SEARCH.

test_params_ = {
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}
# eta (LR) chosen to be 0.3. gamma chosen to be >= 10.
# 'learning_rate': [0.2, 0.3, 0.4],
# 'gamma': [0.1, 1, 10]

n_rounds_: int = 10
base_clf_ = XGBClassifier(objective='binary:logistic', n_estimators=n_rounds_,
                          max_depth='15', num_parallel_tree=50,
                          learning_rate=0.3, gamma=10)
# TODO: UNCOMMENT
# cv_model_ = GridSearchCV(estimator=base_clf_, param_grid=test_params_)
# cv_model_.fit(X_train_, y_train_)
# xgb_bst_ = cv_model_.best_estimator_
#
# print(cv_model_.cv_results_)
# print(f"{cv_model_.best_params_} -> {cv_model_.best_score_}")

#% % XGBOOST (GBTREE)

train_X_xgb_, val_X_xgb_, train_y_xgb_, val_y_xgb_ = train_test_split(X_train_, y_train_, test_size=.15)

xgb.config_context(verbosity=0)
xgb_train_ = xgb.DMatrix(train_X_xgb_, label=train_y_xgb_)
xgb_val_ = xgb.DMatrix(val_X_xgb_, label=val_y_xgb_)
xgb_test_ = xgb.DMatrix(X_test_, label=y_test_)
eval_list_ = [(xgb_train_, 'train'), (xgb_val_, 'eval')]
bst_: xgb.Booster
xgb_out_: np.ndarray
xgb_pred_: np.ndarray

# TODO: UNCOMMENT
# bst_ = xgb.train(params_, xgb_train_, n_rounds_, eval_list_)
# xgb_out_ = bst_.predict(xgb_test_)
# xgb_pred_ = np.asarray([int(out_ + 0.5) for out_ in xgb_out_])
# acc_ = accuracy_score(y_test_, xgb_pred_)
# print(f"{acc_:.4f}")

# %% XGBOOST (DART)

dart_params_ = {
    'booster': 'dart',
    'sample_type': 'uniform',
    'normalize_type': 'tree',
    'objective': 'binary:logistic',
    'rate_drop': 0.1,
    'num_parallel_tree': 100
}

# for sample_type_ in ['weighted', 'uniform']:
#     params_['sample_type'] = sample_type_
#     for normalize_type_ in ['tree', 'forest']:
#         params_['normalize_type'] = normalize_type_
#         for rate_drop_ in [0.05, 0.1, 0.15, 0.2, 0.25]:
#             params_['rate_drop'] = rate_drop_

# TODO: UNCOMMENT
# bst_ = xgb.train(dart_params_, xgb_train_, n_rounds_, eval_list_)
# xgb_out_ = bst_.predict(xgb_test_, ntree_limit=n_rounds_)
# xgb_pred_ = np.asarray([int(out_ + 0.5) for out_ in xgb_out_])
# acc_ = accuracy_score(y_test_, xgb_pred_)
# print(f"{acc_:.4f}")

# %% XGBOOST (LINEAR)

# bst_.save_model('bst.model')
# bst_ = xgb.Booster({})
# bst_.load_model('bst.model')
# bst_.predict(xgb_test_, iteration_range=(0, bst_.best_iteration))

linear_params_ = {
    'booster': 'gblinear',
    'objective': 'binary:logistic',
    'num_parallel_tree': 100
}

# TODO: UNCOMMENT
# bst_ = xgb.train(linear_params_, xgb_train_, n_rounds_, eval_list_)
# xgb_out_ = bst_.predict(xgb_test_)
# xgb_pred_ = np.asarray([int(out_ + 0.5) for out_ in xgb_out_])
# acc_ = accuracy_score(y_test_, xgb_pred_)
# print(f"{acc_:.4f}")

# %% NAIVE BAYES

cat_X_train_: np.ndarray = np.asarray(X_train_)
cat_X_test_: np.ndarray = np.asarray(X_test_)
for i, colname_ in enumerate(X_train_.columns):
    enc_ = OrdinalEncoder()
    cat_X_train_[:, i] = enc_.fit_transform(cat_X_train_[:, i].reshape(-1, 1)).reshape(-1)
    cat_X_test_[:, i] = enc_.fit_transform(cat_X_test_[:, i].reshape(-1, 1)).reshape(-1)

for alpha_ in [0, 10, 100, 1000]:
    nb_clf_ = CategoricalNB(alpha=alpha_)
    nb_clf_.fit(cat_X_train_, y_train_)
    train_acc_ = nb_clf_.score(cat_X_train_, y_train_)
    test_acc_ = nb_clf_.score(cat_X_test_, y_test_)
    print(f"Accuracies: {train_acc_:.4f}, {test_acc_:.4f}")

# %% TRAIN YOUR BEST MODEL

X_final_: pd.DataFrame
y_final_: np.ndarray
y_pred_: np.ndarray

X_final_ = test_df_
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=.15)

y_final_ = np.ndarray(X_final_.shape[0])
y_pred_ = np.ndarray(X_test_.shape[0])

params_ = {
    'booster': 'gbtree',
    'verbosity': 0,
    'max_depth': 30,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'num_parallel_tree': 200,
    'gamma': 1,
    'colsample_bytree': 0.9
}

for type_ in coupon_types_:
    index_ = X_['Coupon'] == type_
    X_by_type_ = X_[index_].drop(columns=['Coupon'])
    y_by_type_ = y_[index_]

    index_ = X_train_['Coupon'] == type_
    X_train_by_type_ = X_train_[index_].drop(columns=['Coupon'])
    y_train_by_type_ = y_train_[index_]

    index_ = X_test_['Coupon'] == type_
    X_test_by_type_ = X_test_[index_].drop(columns=['Coupon'])
    y_test_by_type_ = y_test_[index_]

    # X_train_by_type_, X_val_by_type_, y_train_by_type_, y_val_by_type_ = \
    #     train_test_split(X_train_by_type_, y_train_by_type_, test_size=0.15)

    # dmatrix_val_ = xgb.DMatrix(X_val_by_type_, label=y_val_by_type_, enable_categorical=True)
    dmatrix_train_ = xgb.DMatrix(X_train_by_type_, label=y_train_by_type_, enable_categorical=True)
    dmatrix_test_ = xgb.DMatrix(X_test_by_type_, label=y_test_by_type_, enable_categorical=True)

    t0_ = time()
    eval_list_ = [(dmatrix_train_, 'train'), (dmatrix_test_, 'eval')]
    bst_ = xgb.train(params_, dmatrix_train_, 100, eval_list_)
    print(f"XGB time:{time()-t0_: .6f}")

    t0_ = time()
    rf_ = RandomForestClassifier(ccp_alpha=1e-4)
    rf_.fit(pd.get_dummies(X_by_type_), y_by_type_)
    print(f"SKLEARN time:{time()-t0_: .6f}")

    # xgb_out_ = bst_.predict(dmatrix_train_)
    # print(f"{accuracy_score(y_train_by_type_, [int(out_ + 0.5) for out_ in xgb_out_]): .4f}")
    # xgb_out_ = bst_.predict(dmatrix_test_)
    # print(f"{accuracy_score(y_test_by_type_, [int(out_ + 0.5) for out_ in xgb_out_]):.4f}")
    index_ = X_test_['Coupon'] == type_
    y_pred_[index_] = bst_.predict(xgb.DMatrix(X_test_.drop(columns=['Coupon']), enable_categorical=True))[index_]
    y_pred_[index_] += rf_.predict(pd.get_dummies(X_test_.drop(columns=['Coupon'])))[index_]

    index_ = X_final_['Coupon'] == type_
    y_final_[index_] = bst_.predict(xgb.DMatrix(X_final_.drop(columns=['Coupon']), enable_categorical=True))[index_]
    y_final_[index_] += rf_.predict(pd.get_dummies(X_final_.drop(columns='Coupon')))[index_]

y_pred_ = np.asarray([e/2 for e in y_pred_])
y_final_ = np.asarray([e/2 for e in y_final_])
val_acc_: float = accuracy_score(y_test_, [int(pred_ + 0.5) for pred_ in y_pred_])
print(f"Final Validation Accuracy: {val_acc_}")

# %% SAVE OUTPUTS.

to_save_ = {'id': [i for i in range(1, len(y_final_)+1)],
            'Decision': [int(out_ + 0.5) for out_ in y_final_]}
save_df_ = pd.DataFrame(data=to_save_)
save_df_.to_csv('./predictions.csv', index=False)
print("SAVED")
