import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer

seed = 42  # random seed
random.seed(seed)
samples = 2000  # num of NUM_SAMPLES
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
    train1_path = '../raw_data/ppd/train1.csv'
    train2_path = '../raw_data/ppd/train2.csv'
    train1 = pd.read_csv(train1_path, encoding='gbk')
    train2 = pd.read_csv(train2_path, encoding='gbk')
    ppd = pd.concat([train1, train2], axis=0)  # (60000, 228)
    ppd = ppd.drop('Idx', axis=1)

    mv = ppd.isnull().sum().sort_values()
    mv = mv[mv > 0]
    mv_rate = mv / len(ppd)
    features_to_drop = mv_rate[mv_rate > 0.1].index
    ppd.drop(features_to_drop, axis=1, inplace=True)
    ppd.dropna(axis=0, inplace=True)
    ppd.drop_duplicates(inplace=True)

    categorical = [col for col in ppd.columns if ppd[col].dtypes == 'object']
    # Label Encoding
    lb = LabelEncoder()
    for col in categorical:
        ppd['o' + col] = lb.fit_transform(ppd[col])
        ppd.drop(col, axis=1, inplace=True)

    # normalization
    scaler_minmax = MinMaxScaler()
    app = pd.DataFrame(scaler_minmax.fit_transform(ppd.drop('target', axis=1)),
                       columns=ppd.drop('target', axis=1).columns)
    app['target'] = ppd.target.values
    ppd = app

    # KBinsDiscretization
    need_discretized = ppd.drop(['target'], axis=1)
    est = KBinsDiscretizer(
        n_bins=256, encode='ordinal', strategy='uniform', subsample=None
    )
    discretized = est.fit_transform(need_discretized)
    ppd = pd.concat(
        [
            pd.DataFrame(discretized, columns=need_discretized.columns),
            ppd.loc[:, ['target']].reset_index(drop=True)
        ], axis=1
    )

    # rename
    ppd = ppd.rename(columns={'target': 'loan_status'})

    # sample
    approved_samples = pd.DataFrame()
    seeds = random.sample(range(1, 100001), num_batches)
    for s in seeds:
        df = take_balanced_samples(ppd, num_samples=samples, seed=s)
        approved_samples = pd.concat([approved_samples, df], axis=0)

    return approved_samples


if __name__ == "__main__":
    ppd_df = data_prepro()
    ppd_df.to_csv("../preprocessed/ppd1.csv", index=False)
