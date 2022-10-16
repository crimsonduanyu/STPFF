# -*- coding: utf-8 -*-            
# @Time : 2022/10/15 20:22
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: citibike_datacleasing.py
# @Software: PyCharm

import pandas as pd
file_list = [
    '202204-citibike-tripdata.csv',
    '202205-citibike-tripdata.csv',
    '202206-citibike-tripdata.csv',
    '202207-citibike-tripdata.csv',
    '202208-citibike-tripdata.csv',
    '202209-citibike-tripdata.csv']


region_name = 'Manhattan'


usable_data = pd.DataFrame(columns=['ride_id','rideable_type',
                     'started_at','ended_at',
                     'start_station_name','start_station_id',
                     'end_station_name','end_station_id',
                     'start_lat','start_lng',
                     'end_lat','end_lng',
                     'member_casual'])

from tqdm import tqdm

for csv_file in file_list:
    df = pd.read_csv(csv_file,low_memory=False)
    df = df.dropna(axis=0,how='any')
    with tqdm(total=len(df)) as bar:
        for index,row in df.iterrows():
            bar.update(1)
            if region_name in row[4] or region_name in row[6]:
                #print(row)
                usable_data.loc[len(usable_data)] = row
                pass
            pass
    pass



usable_data.to_csv('usable_data.csv')

print('done')
