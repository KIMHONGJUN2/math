import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
dataset = [['milk', 'cookie', 'apple', 'beans','eggs','yogurt'],
           ['coke', 'cookie','apple','beans','eggs','yogurt'],
           ['milk','apple','kidney beans','eggs',],
           ['milk','orange','corn','beans','yogurt'],
           ['corn','cookie','cookie','beans','ice cream','eggs']]

te= TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary,columns = te.columns_)
print(df)

frequent_itemsets = apriori(df,min_support = 0.6, use_colnames = True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x:len(x))
print(frequent_itemsets)

#자료구조 확인
print(type(dataset))

#트랜잭션 변환
print(te_ary)

#연관규칙 필터링
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x:len(x))
print(frequent_itemsets)
print(frequent_itemsets[(frequent_itemsets['length']==2) & (frequent_itemsets['support'] >= 0.6)]) #2개인 아이템과 지지도 60%이상
print('-------------------------------패턴찾기-----------------------------')
rules = association_rules(frequent_itemsets,metric = 'confidence',min_threshold = 0.7)
print(rules)
rules2 = association_rules(frequent_itemsets,metric='lift',min_threshold=1.2)
print(rules2)
rules['antecedent_len'] = rules['antecedents'].apply(lambda x:len(x))
print(rules[(rules['antecedent_len'] >= 2) & (rules['confidence'] > 0.75) & (rules['lift'] > 1.2)])