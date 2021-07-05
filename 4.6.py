import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import mlxtend
from mlxtend.frequent_patterns import apriori
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
print(frequent_itemsets)