import pandas as pd
import numpy as np
print(pd.__version__)
s = pd.Series([0,0.25,0.5,0.75,1.0] ,index=['a','b','c','d','e'])
print('b' in s)
print(s[2:])
print(s.value_counts())
print(s.isin([0.25,0.75]))
pop_tuple = {'서울특별시': 9720846,
             '부산광역시': 3434242,
             '인천광역시': 343214}
population = pd.Series(pop_tuple)
print(population['서울특별시'])

#dataframe
pd.DataFrame([{'A':2, 'B':4,'D':3} , {'A':4,'B':5,'C':7}] )
print(pd.DataFrame([{'A':2, 'B':4,'D':3} , {'A':4,'B':5,'C':7}] ))
#pd.DataFrame(np.random.randint(5,5),columns=['A','B','C','D','E'])
male_tuple = {'서울특별시': 4732275,
             '부산광역시': 1668618,
             '인천광역시': 1476813}
male = pd.Series(male_tuple)
print(male)
female_tuple = {'서울특별시': 2373575,
             '부산광역시': 2848618,
             '인천광역시': 8736813}
female = pd.Series(female_tuple)
korea_df = pd.DataFrame({'인구수': population,
                         '남자인구수': male,
                         '여자인구수': female})
print(korea_df)

idx = pd.Index([2,4,6,8,10])
print(idx)
print(idx[::2])
print(idx.size)
print(idx.ndim)
print(idx.dtype)
idx1 = pd.Index([1,2,4,6,8])
idx2 = pd.Index([2,4,5,6,7])
print(idx1.append(idx2))
print(idx1.difference(idx2))
print(idx1 - idx2)
print(idx1.intersection(idx2))
print(idx1.union(idx2))
print(idx1 | idx2)
print(idx1.delete(0))
print(idx1.drop(1))

#인덱싱
s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],index=['a','b','c','d','e'])
print(s.keys())
print(list(s.items()))
s['f'] = 1.25
print(s)
print(s[(s>0.4) & (s<0.8)])

print(korea_df.남자인구수)
korea_df['남여비율'] = (korea_df['남자인구수'] * 100 / korea_df['여자인구수'])
print(korea_df.남여비율)
print(korea_df.T)

korea_df
idx_tuple = [('서울특별시',2010), ('서울특별시' , 2020),

             ('부산광역시',2010), ('부산광역시' , 2020),

             ('인천광역시',2010), ('인천광역시' , 2020),

             ('대구광역시',2010), ('대구광역시' , 2020),

             ('대전광역시',2010), ('대전광역시' , 2020)
             ]
print(idx_tuple)
pop_tuples = [10312545,9720846,2567910,3404434,2758296,2947964,2511676,2427954,1503664,1471040]
population = pd.Series(pop_tuples,index=idx_tuple)
print(population)
midx = pd.MultiIndex.from_tuples(idx_tuple)
print(midx)
population = population.reindex(midx)
print(population)
print(population[:,2010])
korea_mdf = population.unstack()
print(korea_mdf)
#korea_mdf = korea_mdf.stack()
print(korea_mdf)
male_tuple2 = [511259, 47332775,
               1773170, 1668618,
               1390356, 1476813,
               1255245, 1198815,
               753648, 734441]
print(male_tuple2)
korea_mdf = pd.DataFrame({'총인구수':population,
                         '남자인구수': male_tuple2})
print(korea_mdf)
female_tuple2 = [5202286, 4988571,
                 1794750, 1735805,
                 1367940, 1470040,
                 1245431, 1229139,
                 750016, 735894]
korea_mdf = pd.DataFrame({'총인구수':population,
                         '남자인구수': male_tuple2,
                          '여자인구수':female_tuple2})
ratio = korea_mdf['남자인구수'] *100/korea_mdf['여자인구수']
print(ratio.unstack())
korea_mdf = pd.DataFrame({'총인구수':population,
                         '남자인구수': male_tuple2,
                          '여자인구수':female_tuple2,
                          '남여비율': ratio})

print(korea_mdf)
df = pd.DataFrame
