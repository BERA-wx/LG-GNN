import random
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')


'''Utils'''


seed = 42
random.seed(seed)
samples = 1000
num_batches = 10


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
    shuffled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df


def process_data(data, year):
    # YEAR
    data['issue_d'] = pd.to_datetime(data['issue_d']).dt.YEAR
    data = data[data['issue_d'] == year]

    # fico_score
    data['fico_score'] = (data['fico_range_high'] + data['fico_range_low']) / 2
    drop = ['fico_range_high', 'fico_range_low']
    data = data.drop(drop, axis=1)

    # loan_status
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    data = data[data['loan_status'].isin(valid_statuses)]
    map_status = {'Fully Paid': 0, 'Default': 1, 'Charged Off': 1}
    data['loan_status'] = data['loan_status'].map(map_status)

    # emp_length: label encoding
    map_emp = {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5,
               '4 years': 4, '3 years': 3, '2 years': 2, '1 YEAR': 1, '< 1 YEAR': 0}
    data['emp_length'] = data['emp_length'].map(map_emp)

    # features
    keep = ['fico_score', 'dti', 'loan_amnt', 'emp_length', 'loan_status']
    data = data[keep]

    # na
    data.dropna(axis=0, how='any', inplace=True)
    # duplicate
    data.drop_duplicates(inplace=True)

    # KBinsDiscretization
    need_discretized = data.drop(['emp_length', 'loan_status'], axis=1)
    est = KBinsDiscretizer(
        n_bins=32, encode='ordinal', strategy='uniform', subsample=None
    )
    discretized = est.fit_transform(need_discretized)
    data = pd.concat(
        [
            pd.DataFrame(discretized, columns=need_discretized.columns),
            data.loc[:, ['emp_length', 'loan_status']].reset_index(drop=True)
        ], axis=1
    )

    # sample
    approved_samples = pd.DataFrame()
    seeds = random.sample(range(1, 10001), num_batches)
    for s in seeds:
        df = take_balanced_samples(data, num_samples=samples, seed=s)
        approved_samples = pd.concat([approved_samples, df], axis=0)

    return approved_samples


if __name__ == '__main__':
    year = 2013

    # 1 import approved NUM_SAMPLES: (2260701, 151) 2007-2018
    df = pd.read_csv('../raw_data/LendingClub/accepted_2007_to_2018Q4.csv')

    # 2 Load approved NUM_SAMPLES
    if year == 2013:
        dn_df = process_data(df, year)
        dn_df.to_csv('../preprocessed/Lending1.csv', index=False)
    elif year == 2014:
        dn_df = process_data(df, year)
        dn_df.to_csv('../preprocessed/Lending2.csv', index=False)
    elif year == 2015:
        dn_df = process_data(df, year)
        dn_df.to_csv('../preprocessed/Lending3.csv', index=False)
