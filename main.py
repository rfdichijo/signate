import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


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
    cols = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # df['High_Glucose'] = df['Glucose'] >= 155
    for col in cols:
        df[col].replace(0, np.nan, inplace=True)
        # df.loc[df[col].isna(), col] = df[col].median()

    # 標準化
    scaler = StandardScaler()
    scaler.fit(df[cols])
    df_sc = scaler.transform(df[cols])
    df_sc = pd.DataFrame(df_sc, columns=cols)
    df_sc2 = df_sc.copy()

    df_sc2.loc[df_sc2['BloodPressure'].isna(), 'BloodPressure'] = df_sc['BMI']
    df_sc2.loc[df_sc2['SkinThickness'].isna(), 'SkinThickness'] = df_sc['Insulin']
    df_sc2.loc[df_sc2['Insulin'].isna(), 'Insulin'] = df_sc['SkinThickness']
    df[cols] = df_sc2[cols]

    return df

def lightgbm_train():
    df_train = get_train_data()
    df_train = engineering(df_train)
    X = df_train.drop(columns='Outcome')
    y = df_train['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)

    # 訓練・テストデータの設定
    train_data = lgb.Dataset(X_train, label=y_train)
    eval_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    # 訓練・テストデータの設定
    train_data = lgb.Dataset(X_train, label=y_train)
    eval_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

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
        'seed': 2021,
        'verbose': 2,
    }
    bst = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=100,
        early_stopping_rounds=20,
        valid_sets=[eval_data],
    )
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred = y_pred.round(0)
    acc = accuracy_score(y_pred, y_test)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)
    print(acc)


if __name__ == '__main__':
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 200)

    lightgbm_train()
    corr()
    get_info(get_train_data())