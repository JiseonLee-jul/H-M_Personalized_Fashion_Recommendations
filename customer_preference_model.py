data.head()

data.t_dat = pd.to_datetime(transactions['t_dat'])
data.t_dat.max()

# submission date(2020-09-23 ~ 2020-09-29) 1, 2, 3주 전 거래내역
data_3w = data[data['t_dat'] >= pd.to_datetime('2020-09-02')].copy()
data_2w = data[data['t_dat'] >= pd.to_datetime('2020-09-09')].copy()
data_1w = data[data['t_dat'] >= pd.to_datetime('2020-09-16')].copy()

# transaction log beofre 3 weeks ago
purchase_dict_3w = {}

for i,x in enumerate(zip(data_3w['customer_id'], data_3w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_3w:
        purchase_dict_3w[cust_id] = {}
    
    if art_id not in purchase_dict_3w[cust_id]:
        purchase_dict_3w[cust_id][art_id] = 0
    
    purchase_dict_3w[cust_id][art_id] += 1
    
print(len(purchase_dict_3w))

dummy_list_3w = list((data_3w['article_id'].value_counts()).index)[:12]

# load submission
submission = pd.read_csv(path + 'sample_submission.csv')
submission["customer_id"] = submission["customer_id"].map(id_to_index_dict)
submission.head()

## model
### 고려할 세 가지 변수
#1. whole_prod_type : product type
#2. perceived_colour_value_name 
#3. graphical_appearance_name

### 추천 우선순위 (기간 : 3주) (2020.09.02~2020.09.22)
#1. 고객별로 가장 많이 구매한 제품군, 색, 프린트가 같은 제품 중 best seller 추천
#2. 고객별로 가장 많이 구매한 색, 프린트가 같은 제품 중 best seller 추천
#3. 고객별로 본인이 가장 자주 구매한 아이템 추천
#4. 1주 전의 best seller 추천  
# 전 주의 베스트 셀러 아이템
best_seller_1w_list = list(data_1w['article_id'].value_counts().index)[:12]
best_seller_1w = ' '.join(best_seller_1w_list)

for i in submission['customer_id']:
    
    if i in purchase_dict_3w:        
        # 1. Recommend best prod type & colour & graphical appearance
                           # customer's best product type
        best_3w = data_3w[(data_3w.whole_prod_type == \
                           data_3w[data_3w['customer_id'] == i][['whole_prod_type']].value_counts().index[0]) &
                           # customer's best colour
                           (data_3w.perceived_colour_value_name == \
                           data_3w[data_3w['customer_id'] == i][['perceived_colour_value_name']].value_counts().index[0]) &
                           # customer's best graphical_appearance
                           data_3w.graphical_appearance_name == \
                           data_3w[data_3w['customer_id'] == i][['graphical_appearance_name']].value_counts().index[0]]
        
        # 1번 조건을 만족하는 데이터가 없음
        if best_3w.empty:
            # 2. Recommend best colour & graphical appearance
                              # customer's best colour
            best_3w = data_3w[(data_3w.perceived_colour_value_name == \
                               data_3w[data_3w['customer_id'] == i][['perceived_colour_value_name']].value_counts().index[0]) &
                               # customer's best graphical_appearance
                               data_3w.graphical_appearance_name == \
                               data_3w[data_3w['customer_id'] == i][['graphical_appearance_name']].value_counts().index[0]]
        
            # 1, 2번 조건 모두 불만족 -> 본인이 구매한 상품 추천
            if best_3w.empty:
                best_3w = data_3w[data_3w.customer_id == i]
                
        # best_3w 완성 후 prediction list(12 articles) 채우기
        # l : 최근 3주 판매량 순서대로 나열한 리스트
        l = list(best_3w.article_id.value_counts().index)
        if len(l) >= 12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l + best_seller_1w_list[:(12 - len(l))])
    
    # 최근 3주 구매 기록이 없는 고객
    else:
        s = best_seller_1w
        
    submission[submission.customer_id == i][['prediction']] = s
    print(i)

submission
