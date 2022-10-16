# -*- coding: utf-8 -*-            
# @Time : 2022/10/15 22:33
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: make_matrix.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from tqdm import tqdm
from math import floor

df = pd.read_csv('destination_Manhattan.csv',low_memory=False)
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

station_dict = df.set_index("end_station_name")["end_station_id"].to_dict()


stat_matrix = pd.DataFrame(np.zeros((8784, 8)), columns=df['end_station_name'].unique())    # 15811200 / 1800 == 8784

df['ended_at'] = df['ended_at'] - pd.to_datetime('2022-04-01 00:00:00')
df['ended_at'] = pd.to_timedelta(df['ended_at'])
df['ended_at'] = df['ended_at'].dt.total_seconds()
total_second = (30 + 31 + 30 + 31 + 31 + 30)*24*3600    # 15811200
half_hr_second = 1800


with tqdm(total=len(df)) as bar:
    for index, line in df.iterrows():
        bar.update(1)
        if line[4] <= 15811200:
            stat_matrix.loc[floor(line[4]//1800)][line[7]] = stat_matrix.loc[floor(line[4]//1800)][line[7]] + 1
        pass



stat_matrix.to_csv('statMatrix.csv')




print('done')
