import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import datetime

# jupyter lab 
!pip install "jupyterlab>=3" "ipywidgets>=7.6"
# jupyter dash 설치
!pip install jupyter-dash


##### load data
# articles dtype 변경 이유 : article_id ex '0108775015' (int로 불러오면 0 사라짐)
path = 'D:/Kaggle/H&M/'
articles = pd.read_csv(path + 'articles.csv', 
                       dtype = {'article_id' : 'object', 'product_code' : 'object'})
customers = pd.read_csv(path + 'customers.csv')
transactions = pd.read_csv( path + 'transactions_train.csv',
                           dtype = {'article_id' : 'object'})


##### data preprocessing
# new variable : whole_prod_type
articles['whole_prod_type'] = articles.index_group_name + "_" + articles.index_name + "_" + articles.product_type_name
# merge data
data = transactions.merge(articles[['article_id', 'whole_prod_type']], 
                          on = 'article_id', how = 'left')

# number of weeks variables
data.t_dat = pd.to_datetime(data.t_dat)
data['num_week'] = data.t_dat.dt.isocalendar().week

# for memory efficiency
id_to_index_dict = dict(zip(customers["customer_id"], customers.index))
index_to_id_dict = dict(zip(data.index, data["customer_id"]))

data["customer_id"] = data["customer_id"].map(id_to_index_dict)

# 2020년 데이터만 사용
data = data[data.t_dat.dt.year == 2020]


##### function 생성
import plotly.express as px

def do_customers_purchase_same_AGGKEY(df, agg_key):
    # dfagg : agg_key에 있는 value들 한 행에 다 넣기(,로 join해서)
    dfagg = df.groupby(['num_week','customer_id'])[[agg_key]].\
               agg({agg_key: lambda x: ','.join(x)}).\
               reset_index().rename(columns={agg_key: 'purchased_set'})
    
    # 새로 생성한 variable(num_{i}wk_before)을 이용한 merge로 중복되는 부분만 남도록 filtering 하기
    # num_{i}wk_before : num_week에서 i주만큼 더한 주의 수(default = 4[4주])
    for i in (range(1, 4 + 1)):
        dfagg[f'num_{i}wk_before'] = dfagg['num_week'] + i
        dfagg = pd.merge(
            dfagg,
            dfagg.rename(columns={'purchased_set': f'{i}wk_before_purchased_set'})\
            [['customer_id',f'num_{i}wk_before',f'{i}wk_before_purchased_set']],
            left_on=['num_week', 'customer_id'],
            right_on=[f'num_{i}wk_before', 'customer_id'],
            how='left'
        )

    # 필요한 column만 선택
    dfagg = dfagg[['num_week','customer_id','purchased_set',\
                   '1wk_before_purchased_set','2wk_before_purchased_set',\
                   '3wk_before_purchased_set','4wk_before_purchased_set']]
    
    # na채우고 ',' 기준 split
    for col in ['purchased_set',\
                   '1wk_before_purchased_set','2wk_before_purchased_set',\
                   '3wk_before_purchased_set','4wk_before_purchased_set']:
        dfagg[col] = dfagg[col].fillna('')
        dfagg[col] = dfagg[col].str.split(',')
    
    # 2~4 주는 전 주의 판매 아이템도 모두 포함해야함
    for i in (range(2, 4 + 1)):
        dfagg[f'{i}wk_before_purchased_set'] = \
            dfagg[f'{i}wk_before_purchased_set'] + dfagg[f'{i-1}wk_before_purchased_set']
    
    # 집합형으로 변환
    for col in ['purchased_set',\
                   '1wk_before_purchased_set','2wk_before_purchased_set',\
                   '3wk_before_purchased_set','4wk_before_purchased_set']:
        dfagg[col] = dfagg[col].map(set)

    # is_purchased_same_within_{i}wk : i주 안에 구매한게 참이면 1 거짓이면 0인 column
    for i in (range(1, 4 + 1)):
        dfagg[f'is_purchased_same_within_{i}wk'] = \
        (dfagg['purchased_set'] & dfagg[f'{i}wk_before_purchased_set']).astype(int)
        per = dfagg[dfagg[f'is_purchased_same_within_{i}wk'] == 1]['customer_id'].nunique() / dfagg['customer_id'].nunique() * 100
        print(f'Percentage of customers within {i} weeks is \
              {per} %')
    
    # data for graph
    df_vis = pd.DataFrame({
        'Pediod': ['Within_1wk', 'Within_2wk', 'Within_3wk', 'Within_4wk'],
        'Ratio': [dfagg[dfagg['is_purchased_same_within_1wk'] == 1]['customer_id'].nunique() / dfagg['customer_id'].nunique() * 100,
                  dfagg[dfagg['is_purchased_same_within_2wk'] == 1]['customer_id'].nunique() / dfagg['customer_id'].nunique() * 100,
                  dfagg[dfagg['is_purchased_same_within_3wk'] == 1]['customer_id'].nunique() / dfagg['customer_id'].nunique() * 100,
                  dfagg[dfagg['is_purchased_same_within_4wk'] == 1]['customer_id'].nunique() / dfagg['customer_id'].nunique() * 100]
    })
    
    fig = px.bar(df_vis, x='Pediod', y='Ratio')
    fig.show()
    
    return dfagg
 
# function 
 do_customers_purchase_same_AGGKEY(data, 'whole_prod_type')
