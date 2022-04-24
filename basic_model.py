import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import datetime

# articles dtype 변경 이유 : article_id ex '0108775015' (int로 불러오면 0 사라짐)
sample_submission = pd.read_csv('D:/Kaggle/H&M/sample_submission.csv')
train = pd.read_csv('D:/Kaggle/H&M/transactions_train.csv',
                    dtype = {'article_id' : 'object'})

# Recent 7 days list
numdays = 7
train.t_dat = pd.to_datetime(train.t_dat)
base = train.t_dat.max()
date_list = [base - datetime.timedelta(days = x) for x in range(numdays)]
print(date_list)

# data for recent 7 days
# 12 best selling articles
data_days7 = train.query("t_dat >= '2020-09-16'")
pred_list = data_days7.article_id.value_counts()[:12].index
pred_str = " ".join(pred_list)
print(pred_str)

# result to_csv
sample_submission.prediction = pred_str
sample_submission.to_csv('D:/Kaggle/H&M/basic_model.csv', index = False)
