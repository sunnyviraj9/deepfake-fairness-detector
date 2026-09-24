import os
import pandas as pd
import numpy as np

def generate_csvs():
    df_train_old = pd.read_csv('dataset/train_15k.csv')
    df_test_old = pd.read_csv('dataset/test_15k.csv')

    pool = pd.concat([df_train_old, df_test_old], ignore_index=True).drop_duplicates(subset=['image_path'])
    print(f'Total available verified image pool: {len(pool)}')
    print('By target:\n', pool['target'].value_counts())

    real_pool = pool[pool['target'] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    fake_pool = pool[pool['target'] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Test: 2,500 real + 2,500 fake = 5,000
    # Train: 12,500 real + 12,500 fake = 25,000
    test_real = real_pool.iloc[:2500]
    test_fake = fake_pool.iloc[:2500]
    test_df = pd.concat([test_real, test_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_real = real_pool.iloc[2500:2500+12500]
    train_fake = fake_pool.iloc[2500:2500+12500]
    train_df = pd.concat([train_real, train_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f'New Train shape: {train_df.shape}')
    print(train_df['target'].value_counts())
    print(f'New Test shape: {test_df.shape}')
    print(test_df['target'].value_counts())

    overlap = set(train_df['image_path']).intersection(set(test_df['image_path']))
    print(f'Train/Test overlap count: {len(overlap)}')

    # Save CSVs
    train_df.to_csv('dataset/train_25k.csv', index=False)
    test_df.to_csv('dataset/test_5k.csv', index=False)
    print('Saved dataset/train_25k.csv and dataset/test_5k.csv!')

if __name__ == '__main__':
    generate_csvs()
