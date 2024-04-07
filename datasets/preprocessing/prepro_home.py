import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

'''Utils'''

seed = 42  # random seed
random.seed(seed)
samples = 2000  # num of samples
num_batches = 10  # num of batches


def take_imbalanced_samples(df, num_samples):
    label_counts = df['loan_status'].value_counts()
    sample_ratio = label_counts / label_counts.sum()
    num_samples_per_label = (sample_ratio * num_samples).round().astype(int)
    sampled_df = df.groupby('loan_status').apply(
        lambda x: x.sample(n=num_samples_per_label[x.name], random_state=seed)).reset_index(drop=True)
    return sampled_df


def take_balanced_samples(df, num_samples, seed):
    unique_labels = df['loan_status'].unique()
    num_samples_per_label = num_samples // len(unique_labels)
    sampled_data = []
    for label in unique_labels:
        label_data = df[df['loan_status'] == label]
        sampled_label_data = label_data.sample(n=num_samples_per_label, random_state=seed)
        sampled_data.append(sampled_label_data)
    sampled_df = pd.concat(sampled_data, ignore_index=True)
    # shuffle
    shuffled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df


def data_prepro():
    # Load data
    app_train_path = '../raw_data/home/application.csv'
    app_train = pd.read_csv(app_train_path)
    # Delete id
    app_train = app_train.drop('SK_ID_CURR', axis=1)

    # Missing values
    mv = app_train.isnull().sum().sort_values()
    mv = mv[mv > 0]
    mv_rate = mv / len(app_train)
    features_to_drop = mv_rate[mv_rate > 0.47].index
    app_train = app_train.drop(features_to_drop, axis=1)

    dict1 = {'Accountants': 1,
             'High skill tech staff': 2, 'Managers': 2, 'Core staff': 2,
             'HR staff': 2, 'IT staff': 2, 'Private service staff': 2, 'Medicine staff': 2,
             'Secretaries': 2, 'Realty agents': 2,
             'Cleaning staff': 3, 'Sales staff': 3, 'Cooking staff': 3, 'Laborers': 3,
             'Security staff': 3, 'Waiters/barmen staff': 3, 'Drivers': 3,
             'Low-skill Laborers': 4}
    app_train['oOCCUPATION_TYPE'] = app_train['OCCUPATION_TYPE'].map(dict1)
    app_train.drop('OCCUPATION_TYPE', axis=1, inplace=True)
    # ORGANIZATION_TYPE
    dict1 = {'Trade: type 4': 1, 'Industry: type 12': 1, 'Transport: type 1': 1, 'Trade: type 6': 1,
             'Security Ministries': 1, 'University': 1, 'Police': 1, 'Military': 1, 'Bank': 1, 'XNA': 1,

             'Culture': 2, 'Insurance': 2, 'Religion': 2, 'School': 2, 'Trade: type 5': 2, 'Hotel': 2,
             'Industry: type 10': 2,
             'Medicine': 2, 'Services': 2, 'Electricity': 2, 'Industry: type 9': 2, 'Industry: type 5': 2,
             'Government': 2,
             'Trade: type 2': 2, 'Kindergarten': 2, 'Emergency': 2, 'Industry: type 6': 2, 'Industry: type 2': 2,
             'Telecom': 2,

             'Other': 3, 'Transport: type 2': 3, 'Legal Services': 3, 'Housing': 3, 'Industry: type 7': 3,
             'Business Entity Type 1': 3,
             'Advertising': 3, 'Postal': 3, 'Business Entity Type 2': 3, 'Industry: type 11': 3, 'Trade: type 1': 3,
             'Mobile': 3,

             'Transport: type 4': 4, 'Business Entity Type 3': 4, 'Trade: type 7': 4, 'Security': 4,
             'Industry: type 4': 4,

             'Self-employed': 5, 'Trade: type 3': 5, 'Agriculture': 5, 'Realtor': 5, 'Industry: type 3': 5,
             'Industry: type 1': 5,
             'Cleaning': 5, 'Construction': 5, 'Restaurant': 5, 'Industry: type 8': 5, 'Industry: type 13': 5,
             'Transport: type 3': 5}
    app_train['oORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].map(dict1)
    app_train.drop('ORGANIZATION_TYPE', axis=1, inplace=True)
    # Label Encoding
    categorical = [col for col in app_train.columns if app_train[col].dtypes == 'object']
    lb = LabelEncoder()
    for col in categorical:
        app_train['o' + col] = lb.fit_transform(app_train[col])
        app_train.drop(col, axis=1, inplace=True)

    # Outliers
    app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # # Feature selection (home2)
    # rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=seed)
    # rf.fit(app_train.drop(['TARGET'], axis=1), app_train.TARGET)
    # # importance
    # feature_importance = rf.feature_importances_
    # features = app_train.drop(['TARGET'], axis=1).columns.values
    # importance_dict = {feature: importance for feature, importance in zip(features, feature_importance)}
    # sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    # # importance ranking
    # print("Feature Importance Ranking:")
    # for rank, (feature, importance) in enumerate(sorted_importance, start=1):
    #     print(f"Rank {rank}: {feature} - Importance: {importance}")
    # # the 20 most important features
    # top_features = [feature for feature, _ in sorted_importance[:20]]
    # print("Top 20 Features:")
    # for feature in top_features:
    #     print(feature)

    # NA
    app_train.dropna(inplace=True)
    # Deduplicate
    app_train.drop_duplicates(inplace=True)

    # KBinsDiscretization
    need_discretized = app_train.drop(['oOCCUPATION_TYPE', 'oNAME_EDUCATION_TYPE', 'TARGET'], axis=1)
    est = KBinsDiscretizer(
        n_bins=256, encode='ordinal', strategy='uniform', subsample=None
    )
    discretized = est.fit_transform(need_discretized)
    app_train = pd.concat(
        [
            pd.DataFrame(discretized, columns=need_discretized.columns),
            app_train.loc[:, ['oOCCUPATION_TYPE', 'oNAME_EDUCATION_TYPE', 'TARGET']].reset_index(drop=True)
        ], axis=1
    )

    # Rename
    app_train = app_train.rename(columns={'TARGET': 'loan_status'})

    # sample
    approved_samples = pd.DataFrame()
    seeds = random.sample(range(1, 100001), num_batches)
    for s in seeds:
        df = take_balanced_samples(app_train, num_samples=samples, seed=s)
        approved_samples = pd.concat([approved_samples, df], axis=0)

    return approved_samples


if __name__ == "__main__":
    home_df = data_prepro()
    home_df.to_csv("../preprocessed/home1.csv", index=False)