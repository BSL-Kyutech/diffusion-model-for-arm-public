from train_data_related import gen_data
import pandas as pd
from tqdm import tqdm
import random
import os

# 教師データを生成してdata/train.csvに保存する
if __name__ == '__main__':
    df = pd.DataFrame()
    for i in tqdm(range(100000)):
        df = pd.concat(
            [df, gen_data(random.randint(300, 980), random.randint(100, 600))])

    if not os.path.exists('data'):
        os.mkdir('data')
    df.to_csv('data/train.csv', index=False)
