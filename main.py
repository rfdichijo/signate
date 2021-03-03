import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_train_data():
    df = pd.read_csv('downloads/train.csv', index_col='index')
    return df

def get_info(df):
    print(df.info())
    df.replace(0, np.nan, inplace=True)
    print(df.isna().sum())

def visualize():
    cols = ['SkinThickness', 'Insulin']
    df = get_train_data()
    print(df.columns)
    for i in df.Outcome.unique():
        df_tmp = df[df.Outcome==i]
        plt.scatter(df_tmp[cols[0]], df_tmp[cols[1]], label=i)
    plt.legend()
    plt.show()

def corr():
    df = get_train_data()
    coef = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']].corr()
    print(coef)


def engineering(df):
    # cols = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # df['High_Glucose'] = df['Glucose'] >= 155
    # for col in cols:
    #     df[col].replace(0, np.nan, inplace=True)
        # df.loc[df[col].isna(), col] = df[col].median()
    df.replace(0, np.nan, inplace=True)

    # 標準化
    scaler = StandardScaler()
    scaler.fit(df)
    df_sc = scaler.transform(df)
    df_sc = pd.DataFrame(df_sc, columns=df.columns, index=df.index)
    df_sc2 = df_sc.copy()
    # 相関が近いもので穴埋めする
    df_sc2.loc[df_sc['BloodPressure'].isna(), 'BloodPressure'] = df_sc['BMI']
    df_sc2.loc[df_sc['SkinThickness'].isna(), 'SkinThickness'] = df_sc['Insulin']
    df_sc2.loc[df_sc['Insulin'].isna(), 'Insulin'] = df_sc['SkinThickness']
    df_sc2.fillna(df_sc2.median(), inplace=True)

    return df_sc2



def preprocessing():
    df_train = get_train_data()
    X = df_train.drop(columns='Outcome')
    y = df_train['Outcome']
    X = engineering(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)

    return X_train, X_test, y_train, y_test

def lightgbm_train(X_train, X_test, y_train):
    # 訓練・テストデータの設定
    train_data = lgb.Dataset(X_train, label=y_train)
    # 訓練・テストデータの設定
    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'num_leaves': 31,
        # # 'min_data_in_leaf': 10,
        # 'max_depth': 100,
        # 'max_bin': 1024,
        # 'learning_rate': 0.01,
        # 'num_iterations': 1000,
        'seed': SEED,
        'verbose': 0,
    }
    bst = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=100,
    )
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred = y_pred.round(0).astype(int)

    return y_pred

def sgd_train(X_train, X_test, y_train):
    sgd_clf = SGDClassifier(random_state=SEED)
    sgd_clf.fit(X_train, y_train)

    return sgd_clf.predict(X_test)

def log_train(X_train, X_test, y_train):
    log_reg = LogisticRegression(random_state=SEED)
    log_reg.fit(X_train, y_train)

    return log_reg.predict(X_test)

def svm_train(X_train, X_test, y_train):
    svm = LinearSVC(random_state=SEED)
    svm.fit(X_train, y_train)

    return svm.predict(X_test)


def decision_train(X_train, X_test, y_train):
    dec_tree = DecisionTreeClassifier(random_state=SEED)
    dec_tree.fit(X_train, y_train)

    return dec_tree.predict(X_test)

def rand_tree_train(X_train, X_test, y_train):
    rand_tree = RandomForestClassifier(random_state=SEED)
    rand_tree.fit(X_train, y_train)

    return rand_tree.predict(X_test)


def make_acc(X_train, X_test, y_train, y_test=None, verbose=0):
    sgd = sgd_train(X_train, X_test, y_train)
    light = lightgbm_train(X_train, X_test, y_train)
    logreg = log_train(X_train, X_test, y_train)
    svm = svm_train(X_train, X_test, y_train)
    dec_tree = decision_train(X_train, X_test, y_train)
    rand_tree = rand_tree_train(X_train, X_test, y_train)

    sgd = pd.Series(sgd, name='sgd')
    light = pd.Series(light, name='light')
    logreg = pd.Series(logreg, name='logreg')
    svm = pd.Series(svm, name='svm')
    dec_tree = pd.Series(dec_tree, name='dec_tree')
    rand_tree = pd.Series(rand_tree, name='rand_tree')
    result = pd.concat([sgd, light, logreg, svm, dec_tree, rand_tree], axis=1)

    y_pred = result.mean(axis=1)
    y_pred = (y_pred >= 0.5).astype(int)

    if y_test is not None:
        y_pred = y_pred.values
        y_test = y_test.values

        if verbose==1:
            print(f'sgd score: {accuracy_score(sgd, y_test)}')
            print(f'lightgbm score: {accuracy_score(light, y_test)}')
            print(f'logistic score: {accuracy_score(logreg, y_test)}')
            print(f'svm score: {accuracy_score(svm, y_test)}')
            print(f'decision_tree score: {accuracy_score(dec_tree, y_test)}')
            print(f'rand_tree score: {accuracy_score(rand_tree, y_test)}')

        return f'voting score: {accuracy_score(y_pred, y_test)}'

    else:
        return y_pred


def make_submit():
    df = get_train_data()
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    X = engineering(X)
    test = pd.read_csv('downloads/test.csv', index_col='index')
    test = engineering(test)
    submit = make_acc(X, test, y)
    # submit = lightgbm_train(X, test, y)
    ser_index = pd.Series(test.index, name='index')
    ser_sub = pd.Series(submit, name='submit')
    submit = pd.concat([ser_index, ser_sub], axis=1)

    return submit

if __name__ == '__main__':
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 200)

    SEED = 2021
    X_train, X_test, y_train, y_test = preprocessing()
    output = make_acc(X_train, X_test, y_train, y_test)
    print(output)
    submit = make_submit()
    submit.to_csv('sub.csv', index=False, header=False)