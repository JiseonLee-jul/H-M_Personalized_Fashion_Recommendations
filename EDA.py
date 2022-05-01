import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import datetime as dt

# articles dtype 변경 이유 : article_id ex '0108775015' (int로 불러오면 0 사라짐)
articles = pd.read_csv('D:/Kaggle/H&M/articles.csv', 
                       dtype = {'article_id' : 'object', 'product_code' : 'object'})
customers = pd.read_csv('D:/Kaggle/H&M/customers.csv')
transactions = pd.read_csv('D:/Kaggle/H&M/transactions_train.csv',
                           dtype = {'article_id' : 'object'})

############################ articles
# article_id : 제품 id
# product_code, prod_name : 제품코드, 제품명
# product_type, product_type_name : 제품 카테고리의 코드, 제품 카테고리의 이름
# product_group_name : product_type보다 상위 제품 카테고리
# graphical_appearance_no, graphical_appearance_name : 무늬 코드, 무늬 이름
# colour_group_code, colour_group_name : 색 그룹 코드, 색 그룹 이름
# perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name : 추가 색상 정보
# department_no, department_name: : 모든 dep 코드, 이름
# index_code, index_name: :모든 index 코드, 이름
# index_group_no, index_group_name: : index 그룹 코드, 이름
# section_no, section_name: : 각 section의 코드, 이름
# garment_group_no, garment_group_name: : 각 의류 코드, 이름
# detail_desc: : 상세설명

articles.head()
articles.info()

# num_unique : data column의 nunique를 한꺼번에 print하는 함수
def num_unique(data):
    for col in data.columns:
        n = data[col].nunique()
        print(f'{col} : {n}')

# product_type, deparment, section 변수 no랑 name 숫자가 안 맞음
# 코드 number가 더 많은걸 봐서 name에 중복으로 들어간게 있을수도
num_unique(articles)

# num_na : data column의 na 개수, 비율 print하는 함수
def num_na(data):
    for col in data.columns:
        n = data[col].isnull().sum() # na 개수
        per = (n / data[col].shape[0]) * 100 # na 비율
        print(f'{col} : (na 개수 : {n}) (na 비율 : {per})')

# detail_desc : 426 na
num_na(articles)

### plot bar function
import matplotlib.ticker as mtick
# plot_bar : sorted horizontal barplots
def plot_bar(database, col, figsize = (13,5), pct=False, label='articles'):
    fig, ax = plt.subplots(figsize = figsize, facecolor='lightgray')
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 백분율
    if pct:
        data = database[col].value_counts()
        data = data.div(data.sum()).mul(100)
        data = data.reset_index()
        ax = sns.barplot(data=data, x=col, y='index', color='royalblue', lw=1.5, ec='black', zorder=2)
        ax.set_xlabel('% of ' + label, fontsize=10, weight='bold')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        data = database[col].value_counts().reset_index()
        ax = sns.barplot(data=data, x=col, y='index', color='royalblue', lw=1.5, ec='black', zorder=2)        
        ax.set_xlabel('# of articles' + label)
        
    ax.grid(zorder=0)
    ax.text(0, -0.75, col, color='black', fontsize=10, ha='left', va='bottom', weight='bold', style='italic')
    ax.set_ylabel('')
        
    plt.show()

# Ladies wear, Baby/Childern이 70% 가량
plot_bar(articles, 'index_group_name', pct = True)
# ladies wear, divided가 상위 두 그룹
plot_bar(articles, 'index_name', pct = True)

# Baby/Childeren, Ladieswear만 세부 분류 있음
# Baby는 주로 사이즈별로, Ladieswear는 주로 종류별로
articles.groupby(['index_group_name', 'index_name']).size()

# divided category 특성 파악을 위한 예시 보기
art_divided = articles[articles['index_group_name'] == 'Divided']
art_divided[['prod_name', 'product_type_name', 'index_name', 'garment_group_name', 'detail_desc']].drop_duplicates().head(10)

plot_bar(articles, 'product_group_name', pct = True)
# product category counts
# 132개의 product type 
# product_type_no와 nunique 같아짐 -> 중복으로 쓰는 type_name 있는듯
articles.groupby(['product_group_name','product_type_name']).size()

# Number of subcategories
for group in articles['product_group_name'].unique():
    n = len(articles.groupby(['product_group_name','product_type_name']).size()[group])
    print(f'{group} : {n}')

plot_bar(articles, 'colour_group_name', figsize = (15, 12), pct = True)
plot_bar(articles, 'perceived_colour_value_name', pct = True)
plot_bar(articles, 'perceived_colour_master_name', pct = True)

# There are three different types of columns regarding to colour feature
# We need to find some differences between columns
from random import sample

def show_images_in_category(column, value, no_imgs = 3):
    data = articles[articles[column] == value]
    cat_ids = data['article_id'].iloc[:no_imgs].to_list()
    
    fig, ax = plt.subplots(1, no_imgs, figsize = (12, 4))
    
    for i, prod_id in enumerate(cat_ids):
        folder = str(prod_id)[:3]
        file_path = f'D:/Kaggle/H&M/images/{folder}/{prod_id}.jpg'
        
        img = plt.imread(file_path)
        ax[i].imshow(img, aspect = 'equal')
        ax[i].grid(False)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
        ax[i].set_xlabel(articles[articles['article_id'] == prod_id]['prod_name'].iloc[0])
        
    fig.suptitle(f'Articles from a {value} category')
    plt.show()

# perceived_colour_value_name : 전체적인 tone
show_images_in_category('perceived_colour_value_name', 'Medium Dusty', 5)
# perceived_colour_master_name
show_images_in_category('perceived_colour_master_name', 'Grey', 5)
# colour_group_name : 가장 세부적인 사항인듯
show_images_in_category('colour_group_name', 'Grey', 5)

# colour_group_name, perceived_colour_value_name, perceived_colour_master_name
def show_images_by_color(group, value, master, no_imgs = 4):
    data = articles[(articles['colour_group_name'] == group) &
                    (articles['perceived_colour_value_name'] == value) &
                    (articles['perceived_colour_master_name'] == master)]
    cat_ids = data['article_id'].iloc[:no_imgs].to_list()
    
    fig, ax = plt.subplots(nrows = 1, ncols = no_imgs, figsize = (12, 4))
    
    for i, prod_id in enumerate(cat_ids):
        folder = prod_id[:3]
        file_path = f'D:/Kaggle/H&M/images/{folder}/{prod_id}.jpg'
    
        img = plt.imread(file_path)
        ax[i].imshow(img, aspect = 'equal')
        ax[i].grid(False)
        ax[i].set_xticks([],[])
        ax[i].set_yticks([],[])
        ax[i].set_xlabel(articles[articles['article_id'] == prod_id]['prod_name'].iloc[0])
        
    fig.suptitle(f'Color group : {group}, Perceived color value : {value}, Perceived color master : {master}')
    plt.show()

articles.groupby(['perceived_colour_value_name','perceived_colour_master_name']).size()


# solid pattern이 거의 절반가량
plot_bar(articles, 'graphical_appearance_name', pct = True)
# solid : 무늬 없음
show_images_in_category('graphical_appearance_name', 'Solid', 5)
show_images_in_category('graphical_appearance_name', 'All over pattern', 5)
show_images_in_category('graphical_appearance_name', 'Melange', 5)

###################################### customers
customers.head()
customers.info()

# FN, Active -> binary feature (NaN에 0 값 채워넣기)
print(customers.FN.unique(), customers.Active.unique())
customers.FN.fillna(0, inplace = True)
customers.Active.fillna(0, inplace = True)
print(customers.FN.unique(), customers.Active.unique())

num_unique(customers)
num_na(customers)

print(customers.FN.value_counts())
print(customers.Active.value_counts())

customers.club_member_status.fillna('NaN', inplace = True)
print(customers.club_member_status.value_counts())
plot_bar(customers, 'club_member_status', pct = True, label = 'customers')

customers.fashion_news_frequency.fillna('NaN', inplace = True)
print(customers.fashion_news_frequency.value_counts())
plot_bar(customers, 'fashion_news_frequency', pct = True, label = 'customers')
# NONE이 다른 방식으로 코딩됨 -> 하나로 합치자

## to see distribution custmoers' age
def plot_hist(database, col, figsize = (13,5), bins = 50, median = False):
    fig, ax = plt.subplots(figsize = figsize, facecolor='lightgray')
    ax = sns.histplot(data = database, x = col, bins = bins, 
                      color = 'orange', stat = 'percent')
    
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('black')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.set_xlabel(f'Distribution of {col}', fontsize=10, weight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.text(12, 5.5, f'Distribution of {col}', color = 'black', fontsize = 10,
            ha = 'left', weight = 'bold', style = 'italic')
    # median line
    if median:
        median = database[col].median()
        ax.axvline(x = median, color = 'red', ls = '--')
        ax.text(median, 3.5, f'median : {median}', rotation = 'vertical', ha = 'right')
        
    plt.show()

plot_hist(customers, 'age', bins = customers['age'].nunique(), median = True)

###################################### transaction
transactions.head()
transactions.info()
num_na(transactions)
num_unique(transactions)

# box plot of price variable
fig, ax = plt.subplots(figsize = (13, 5))
ax = sns.boxplot(data = transactions, x = 'price')
    
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

ax.set_xlabel('Box plot of price', fontsize=10, weight='bold')
plt.show()

print(f'Q1 = {transactions.price.quantile(.25)}, Q3 = {transactions.price.quantile(.75)}')

# We need to detect and remove outliers using several methods.

# datetime type으로 변경
transactions.t_dat = pd.to_datetime(transactions.t_dat)
print(f'Date range is from {transactions.t_dat.min()} to {transactions.t_dat.max()}')

from matplotlib import dates

# Number of transaction per day
t_day_series = transactions.groupby(['t_dat']).size()

fig, ax = plt.subplots(figsize = (13, 5))
ax = sns.lineplot(x = t_day_series.index, y = t_day_series.values, color = 'green')

for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
    
ax.set_xlabel('date')
ax.set_ylabel('Number of transaction')
ax.set_xticks(range(0,len(t_day_series), ))
ax.xaxis.set_major_locator(dates.MonthLocator(interval = 3))
ax.set_xlim(t_day_series.index.min(), t_day_series.index.max())

plt.show()

transactions['year_month'] = transactions.t_dat.dt.to_period('M')
transactions.year_month


month_price = transactions.groupby(['year_month']).mean().price
month_price.index = month_price.index.to_timestamp()

# Mean prices per month
fig, ax = plt.subplots(figsize = (13, 5))
ax = sns.lineplot(x = month_price.index, y = month_price.values, color = 'green')

for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
    
ax.set_xlabel('date')
ax.set_ylabel('Prices')
ax.set_xlim(month_price.index.min(), month_price.index.max())

plt.show()

# 여름 겨울 시즌에 가격이 떨어지는 경향, 가을 봄 시즌 가격이 더 높음
# 아마도 article 자체 가격 때문일수도 있음
# 이후 combined data에서 다뤄볼 문제
