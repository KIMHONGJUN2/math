import pandas as pd
import numpy as np
import timeit
import matplotlib
from datetime import datetime

# print(pd.__version__)
# s = pd.Series([0,0.25,0.5,0.75,1.0] ,index=['a','b','c','d','e'])
# print('b' in s)
# print(s[2:])
# print(s.value_counts())
# print(s.isin([0.25,0.75]))
# pop_tuple = {'서울특별시': 9720846,
#              '부산광역시': 3434242,
#              '인천광역시': 343214}
# population = pd.Series(pop_tuple)
# print(population['서울특별시'])
#
# #dataframe
# pd.DataFrame([{'A':2, 'B':4,'D':3} , {'A':4,'B':5,'C':7}] )
# print(pd.DataFrame([{'A':2, 'B':4,'D':3} , {'A':4,'B':5,'C':7}] ))
# #pd.DataFrame(np.random.randint(5,5),columns=['A','B','C','D','E'])
# male_tuple = {'서울특별시': 4732275,
#              '부산광역시': 1668618,
#              '인천광역시': 1476813}
# male = pd.Series(male_tuple)
# print(male)
# female_tuple = {'서울특별시': 2373575,
#              '부산광역시': 2848618,
#              '인천광역시': 8736813}
# female = pd.Series(female_tuple)
# korea_df = pd.DataFrame({'인구수': population,
#                          '남자인구수': male,
#                          '여자인구수': female})
# print(korea_df)
#
# idx = pd.Index([2,4,6,8,10])
# print(idx)
# print(idx[::2])
# print(idx.size)
# print(idx.ndim)
# print(idx.dtype)
# idx1 = pd.Index([1,2,4,6,8])
# idx2 = pd.Index([2,4,5,6,7])
# print(idx1.append(idx2))
# print(idx1.difference(idx2))
# print(idx1 - idx2)
# print(idx1.intersection(idx2))
# print(idx1.union(idx2))
# print(idx1 | idx2)
# print(idx1.delete(0))
# print(idx1.drop(1))
#
# #인덱싱
# s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],index=['a','b','c','d','e'])
# print(s.keys())
# print(list(s.items()))
# s['f'] = 1.25
# print(s)
# print(s[(s>0.4) & (s<0.8)])
#
# print(korea_df.남자인구수)
# korea_df['남여비율'] = (korea_df['남자인구수'] * 100 / korea_df['여자인구수'])
# print(korea_df.남여비율)
# print(korea_df.T)
#
# korea_df
# idx_tuple = [('서울특별시',2010), ('서울특별시' , 2020),
#
#              ('부산광역시',2010), ('부산광역시' , 2020),
#
#              ('인천광역시',2010), ('인천광역시' , 2020),
#
#              ('대구광역시',2010), ('대구광역시' , 2020),
#
#              ('대전광역시',2010), ('대전광역시' , 2020)
#              ]
# print(idx_tuple)
# pop_tuples = [10312545,9720846,2567910,3404434,2758296,2947964,2511676,2427954,1503664,1471040]
# population = pd.Series(pop_tuples,index=idx_tuple)
# print(population)
# midx = pd.MultiIndex.from_tuples(idx_tuple)
# print(midx)
# population = population.reindex(midx)
# print(population)
# print(population[:,2010])
# korea_mdf = population.unstack()
# print(korea_mdf)
# #korea_mdf = korea_mdf.stack()
# print(korea_mdf)
# male_tuple2 = [511259, 47332775,
#                1773170, 1668618,
#                1390356, 1476813,
#                1255245, 1198815,
#                753648, 734441]
# print(male_tuple2)
# korea_mdf = pd.DataFrame({'총인구수':population,
#                          '남자인구수': male_tuple2})
# print(korea_mdf)
# female_tuple2 = [5202286, 4988571,
#                  1794750, 1735805,
#                  1367940, 1470040,
#                  1245431, 1229139,
#                  750016, 735894]
# korea_mdf = pd.DataFrame({'총인구수':population,
#                          '남자인구수': male_tuple2,
#                           '여자인구수':female_tuple2})
# ratio = korea_mdf['남자인구수'] *100/korea_mdf['여자인구수']
# print(ratio.unstack())
# korea_mdf = pd.DataFrame({'총인구수':population,
#                          '남자인구수': male_tuple2,
#                           '여자인구수':female_tuple2,
#                           '남여비율': ratio})
#
# print(korea_mdf)
# df = pd.DataFrame
#
# #다중 인덱스 생성
# df = pd.DataFrame(np.random.rand(6,3),
#                   index=[['a','a','b','b','c','c'],[1,2,1,2,1,2]],
#                   columns=['c1','c2','c3'])
# print(df)
# pd.MultiIndex.from_arrays([('a1',1),('a2',2),('b',1),('b',2)])
#
# df2 = pd.DataFrame(np.random.randint(0,20,(5,5)),
#                    columns=list('BACDE'))
#
# df1 = pd.DataFrame(np.random.randint(0,20,(3,3)),
#                    columns=list('ACD'))
# print(df2+df1)
# fval = df1.stack().mean()
# print(df1.add(df2,fill_value=(fval)))
# a = np.random.randint(1,10,size=(3,3))
# print(a)
# print(a +a[0])
# s =pd.Series([-2,3,5,1,2,6,74,3])
# print(s.rank())
# norw, ncol = 10000,100
# df1,df2,df3,df4 = (pd.DataFrame(np.random.rand(norw,ncol)) for i in range(4))
# #print(timeit.timeit('df1 +df2 +df3 +df4') )
#
# #데이터 결합
# s1 = pd.Series(['a','b'], index=[1,2])
# s2 = pd.Series(['c','d'], index=[3,4])
# pd.concat([s1,s2])


# # ------------------------------------0731-------------------
# def create_df(cols,idx):
#     data = {c: [str(c.lower()) + str(i) for i in idx] for c in cols}
#     return pd.DataFrame(data,idx)
# df1 = create_df('AB',[1,2])
# df2 = create_df('AB',[3,4])
# print(df1)
# print(pd.concat([df1,df2]))
# df3 = create_df('AB',[0,1])
# df4 = create_df('CD',[0,1])
# print(pd.concat([df3,df4]))
#
# #print(pd.concat([df1,df3],verify_integrity=True))
#
# print(pd.concat([df1,df3],keys=['X','Y']))
#
# df5 = create_df('ABC',[1,2])
# df6 = create_df('BCD',[3,4])
# print(pd.concat([df5,df6]))
#
# print(pd.concat([df5,df6], join='inner'))
# print(df5.append(df6))
#
# print(pd.concat([df1,df3]),axis =1)

#병합과 조인
# df1 = pd.DataFrame({'학생': ['홍길동','이순신','임꺽정','김유신'],
#                     '학과':['경영학과','교육학과','컴퓨터학과','통계학과']})
# df2 = pd.DataFrame({'학생': ['홍길동','이순신','임꺽정','김유신'],
#                    '입학년도':[2012,2016,2019,2020]})
# print(df1)
# print(df2)
# df3 = pd.merge(df1,df2)
# print(df3)
# df4 = pd.DataFrame({ '학과': ['경영학과','교육학과','컴퓨터학과','통계학과'],
#                      '학과장': ['황희','장영실','안창호','정약용']})
# print(pd.merge(df3,df4))
#
# df5 = pd.DataFrame({'학과':['경영학과','교육학과','교육학과','컴퓨터학과','컴퓨터학과','통계학과'],
#                     '과목': ['경영개론','기초수학','물리학','프로그래밍','운영체제','확률론']})
# print(pd.merge(df1,df5))   # 공통기준인 학과로 머지
#
# print(pd.merge(df1,df2,on='학생'))
# df6 = pd.DataFrame({'이름': ['홍길동','이순신','임꺽정','김유신'],
#                     '성적': ['A','A+','B','A+']})
# print(pd.merge(df1,df6,left_on='학생',right_on='이름'))
#
# print(pd.merge(df1,df6,left_on='학생',right_on='이름').drop('이름',axis=1))
# mdf1 = df1.set_index('학생')
# mdf2 = df2.set_index('학생')
#
# print(pd.merge(mdf1,mdf2,left_index=True,right_index=True))
# print(mdf1.join(mdf2))
#
# df7 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정'],
#                     '주문음식': ['햄버거','피자','자장면']})
#
# df8 = pd.DataFrame({'이름':['홍길동','이순신','김유신'],
#                     '주문음료': ['콜라','사이다','커피']})
# print(pd.merge(df7,df8)) #공통된 것만 머지됨
# print(pd.merge(df7,df8,how='outer'))
#
# print(pd.merge(df7,df8,how='left'))
# print(pd.merge(df7,df8,how='right')) #기준을 뭐로 할것이냐
#
# df9 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정','김유신'],
#                     '순위':[3,2,4,1]})
#
# df10 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정','김유신'],
#                     '순위':[4,1,3,2]})
# print(pd.merge(df9,df10,on='이름'))  #동일한 이름이므로 x, y가 붙음 ex_ 순위_x
# print(pd.merge(df9,df10,on='이름',suffixes=['_인기','_성적']))
#


#데이터 집계와 그룹 연산
# df = pd.DataFrame([[1,1.2,np.nan],
#                   [2.4,5.5,4.2],
#                   [np.nan,np.nan,np.nan],
#                   [0.44,-3.1,-4.1]],
#                   index=[1,2,3,4],
#                   columns=['A','B','C'])
# print(df.head(2)) #앞 tail = 뒤
# print(df.describe())
# print(df)
# print(np.argmin(df),np.argmax(df))
# print(df.idxmin())
# print(df.idxmax())
# print(df.var())
# print(df.skew())
# print(df.sum())
# print(df.cumsum()) #누적합
# print(df.cumprod())
# print(df.prod())
#
# print(df.diff())
#
# print(df.quantile())
# print(df.pct_change())
#
# print(df.corr())
# print(df.corrwith(df.B))
# print(df.cov())
#
# print(df['B'].unique())
# print(df['A'].value_counts())

#group by
#
# df = pd.DataFrame({'c1': ['a','a','b','b','c','d','b'],
#                    'c2': ['A','B','B','A','D','C','C'],
#                    'c3': np.random.randint(7),
#                    'c4': np.random.randint(7)})
# print(df.dtypes)
#
# print(df['c3'].groupby(df['c1']).mean())
# print(df['c4'].groupby(df['c2']).std())
# print(df['c4'].groupby([df['c1'],df['c2']]).mean())
# print(df['c4'].groupby([df['c1'],df['c2']]).mean().unstack()) #데이터프레임 형식으로 보기
# print(df.groupby('c1').mean())
#
# print(df.groupby(['c1','c2']).mean())
# print(df.groupby(['c1','c2']).size())
#
# for c1,group in df.groupby('c1'):
#     print(c1)
#     print(group)
#
# for (c1,c2),group in df.groupby(['c1','c2']):
#     print((c1,c2))
#     print(group)
# print(df.groupby(['c1','c2'])[['c4']].mean())
#
# print(df.groupby('c1')['c3'].quantile())
# print(df.groupby(['c1','c2'])['c4'].agg(['mean','min','max']))
# print(df.groupby(['c1','c2'],as_index=False)['c4'].mean())
#
# def top(df,n=3,column = 'c1'):
#     return df.sort_values(by=column)[-n:]
#
# print(top(df,n=5))
#
# print(df.groupby('c1').apply(top))
#
# #피벗 테이블
#
# print(df.pivot_table(['c3','c4'],
#                index=['c1'],
#                columns=['c2']))
#
# print(df.pivot_table(['c3','c4'],
#                index=['c1'],
#                columns=['c2'],
#                      margins=True)) # margins 부분합 추가
#
#
# print(df.pivot_table(['c3','c4'],
#                index=['c1'],
#                columns=['c2'],
#                      margins=True,aggfunc=sum
#                      ,fill_value=0)) #fillvalue nan을 0으로 채움
# print(pd.crosstab(df.c1,df.c2,values=df.c3,aggfunc=sum,margins=True))
#
# #범주형 데이터
# s = pd.Series(['c1','c2','c1','c2','c1'] * 2)
# print(pd.unique(s))
# print(pd.value_counts(s))
# code =pd.Series([0,1,0,1,0] * 2)
# d =pd.Series(['c1','c2'])
#
# print(d.take(code))
#
# df = pd.DataFrame({'id': np.arange(len(s)),
#                    'c': s,
#                    'v': np.random.randint(1000,5000,size=len(s))})
# print(df)
#
# c=df['c'].astype('category')
# print(c)
# print(c.values.codes)
# df['c'] = c
# print(df.c)
#
# c = pd.Categorical(['c1','c2','c3','c1','c2'])
# print(c)
# categories = ['c1','c2','c3']
# codes = [0,1,2,0,1]
# c= pd.Categorical.from_codes(codes,categories)
# print(c)
# c= pd.Categorical.from_codes(codes,categories,ordered=True)
# print(c)
#
# print(c.as_ordered())
#
# print(c.codes,c.categories)
#
# print(c.value_counts())
#
# #문자열 연산
# name_tuple = ['Aaa','Bbb','CCc','DDd',None,'eEe','FFF','Ggg']
# print(name_tuple)
# names = pd.Series(name_tuple)
# print(names)
# print(names.str.lower())
# print(names.str.len()) #문자열과 관련된 것은 .str로 접근
# print(names.str[0:4])
# print(names.str.split().str.get(-1))
# print(names.str.join('*'))
# print(names.str.match('([A-Za-z]+)'))
#
# #시계열 처리  --- pandas 는 금융에서 자주 쓰음 ex_주식
# idx = pd.DatetimeIndex(['2019-01-01','2020-01-01','2020-02-01','2020-02-02','2020-03-01'])
# s = pd.Series([0,1,2,3,4],index=idx)
# print(s)
# print(s['2020-01-01':])
# print(s['2019'])
#
# dates = pd.to_datetime(['12-12-2019',datetime(2020,1,1),'2nd of Feb, 2020','2020-Mar-4','20200701'])
# print(dates)
#
#
#
# print(dates - dates[0])
#
# print(pd.date_range('2020-01-01','2020-07-01'))
#
# print(pd.date_range('2020-01-01',periods=7))
# print(pd.date_range('2020-01-01',periods=7,freq='M'),)
#
# idx = pd.to_datetime(['2020-01-01 12:00:00','2020-01-02 00:00:00'] + [None])
# print(idx)  #시간 값이 아닐 때는 nat 으로 표시된다
#
# print(pd.isnull(idx))
#
# dates = [datetime(2020,1,1), datetime(2020,1,2),datetime(2020,1,4),datetime(2020,1,7),
#          datetime(2020,1,10),datetime(2020,1,11),datetime(2020,1,15)]
# ts = pd.Series(np.random.randn(7),index=dates)
# print(ts.index)
#
# print(ts.index[0])
#
# ts[ts.index[2]]
#
# print(ts['1/4/2020'])
#
# ts = pd.Series(np.random.randn(1000),
#                index=pd.date_range('2017-10-01',periods=1000))
# print(ts)
# print(ts['2020'])
#
# ts['2020-06'] #날짜별로 일별로 인덱싱 가능
#
# ts[datetime(2020,6,20):]
#
# print(ts['2020-06-10':'2020-06-20'])
#
# tdf = pd.DataFrame(np.random.randn(1000,4),
#                    index=pd.date_range('2017-10-01',periods=1000),
#                    columns=['A','B','C','D'])
# print(tdf)
# print(tdf['2020-06-20':])
# print(tdf['C'])

#ts = pd.Series(np.random.randn(10),
#                index=pd.DatetimeIndex(['2020-01-01','2020-01-01','2020-01-02','2020-01-02',
#                                       '2020-01-03','2020-01-04','2020-01-05','2020-01-05',
#                                       '2020-01-06','2020-01-07']))
#
# print(ts.index.is_unique)
#
# print(ts['2020-01-01'])
#
# print(ts.groupby(level=0).mean())
# print(pd.date_range('2020-01-01,','2020-07-01'))
#
# print(pd.date_range(start='2020-01-01',periods=10))
# print(pd.date_range(end='2020-07-01',periods=10))
# print(pd.date_range('2020-07-01','2020-07-07',freq='B'))
#
# print(pd.timedelta_range(0,periods=12,freq='H'))
# print(pd.timedelta_range(0,periods=60,freq='T'))
#
# print(pd.timedelta_range(0,periods=10,freq='1H30T'))
#
# pd.date_range('2020-01-01',periods=20,freq='B')
#
# ts = pd.Series(np.random.randn(5),
#                index=pd.date_range('2020-01-01',periods=5,freq='B'))
#
# print(ts.shift(1))
# print(ts.shift(3))
# print(ts.shift(-2))
#
# print(ts.shift(3,freq='B'))
# print(ts.shift(3,freq='W'))
#
# import pytz
# print(pytz.common_timezones)
#
# tz =pytz.timezone('Asia/Seoul')
# dinx = pd.date_range('2020-01-01 09:00',periods=7,freq='B')
# ts = pd.Series(np.random.randn(len(dinx)),index=dinx)
# print(ts)
#
#
# pd.date_range('2020-01-01 09:00',periods=7,freq='B',tz='UTC')
# ts_utc = ts.tz_localize('UTC')
# print(ts_utc)
# print(ts_utc.tz_convert('Asia/Seoul'))
#
# ts_seoul = ts.tz_localize('Asia/Seoul')
# print(ts_seoul)
# print(ts_seoul.tz_convert('UTC'))
#
# print(ts_seoul.tz_convert('Europe/Berlin'))
#
# stamp_ny = pd.Timestamp('2020-01-01',tz='America/New_York')
# print(stamp_ny.value)
#
# print(stamp_ny.tz_convert('Asia/Shanghai'))
#
# from pandas.tseries.offsets import Hour
#
# stamp = Hour()
#
#
#
# pr = pd.period_range('2020-01-01','2020-06-30', freq='M')
# print(pd.Series(np.random.randn(6),index=pr))
#
# pidx = pd.PeriodIndex(['2020-1','2020-2','2020-4'],freq='M')
#
# p = pd.Period('2020Q2',freq = 'Q-JAN')
# print(p)
#
# print(p.asfreq('D','start'))
# print(p.asfreq('D','end'))
#
# pr = pd.period_range('2019Q3','2020Q3',freq='Q-JAN')
# ts = pd.Series(np.arange(len(pr)),index=pr)
#
# print(ts)
# pr = pd.period_range('2020-01-01',periods=5,freq='Q-JAN')
# ts = pd.Series(np.random.randn(5),index=pr)
#
# print(ts)
#
# pr1 = pd.date_range('2020-01-01',periods=5,freq='D')
# ts1  = pd.Series(np.random.randn(5),index=pr1)
# print(ts1)
#
# p = ts1.to_period('M')
# print(p)
#
# print(p.to_timestamp(how='start'))
#
# #리샘플링 - - 많이 사용
# dr = pd.date_range('2020-01-01',periods=200,freq='D')
# ts = pd.Series(np.random.randn(len(dr)),index=dr)
#
# print(ts.resample('M').mean())
#
# print(ts.resample('M',kind='period').mean())
#
# dr = pd.date_range('2020-01-01',periods=10,freq='T')
# ts = pd.Series(np.arange(10),index=dr)
#
# print(ts)
#
# print(ts.resample('2T',closed='left').sum()) # 동일한 레벨에서 리샘플링
#
# print(ts.resample('2T',closed='right').sum())
#
# print(ts.resample('2T',closed='right',label='right').sum())
#
# print(ts.resample('2T',closed='right',label='right',loffset='-1s').sum()) # -1초 오프셋
#
# print(ts.resample('2T').ohlc())
#
# df = pd.DataFrame(np.random.randn(10,4),
#                   index=pd.date_range('2019-10-01',periods=10,freq='M'),
#                   columns=['C1','C2','C3','C4'])
# print(df)
#
# print(df.resample('Y').asfreq()) # 연도 year 기준으로 리샘플
#
# print(df.resample('W-FRI').asfreq())
# print(df.resample('H').asfreq())
#
# print(df.resample('H').ffill())

#무빙윈도우
df = pd.DataFrame(np.random.randn(300,4),
                  index=pd.date_range('2020-01-01',periods=300,freq='D'),
                  columns=['C1','C2','C3','C4'])
print(df)
print(df.rolling(30).mean())
print(df.rolling(30).mean().plot())
print(df.rolling(60).mean().plot(logy=True))

print(df.C1.rolling(60,min_periods=10).std().plot())

# # 데이터 읽기 및 저장
# %%writefile example1.csv
# a, b, c, d, e, text
# 1, 2, 3, 4, 5, hi
#
# dr = pd.date_range('2020-01-01', periods=10)
# ts = pd.Series(np.arange(10),index=dr)
#
# ts.to_csv('ts.csv',header=['value'])

df = pd.DataFrame({'a': np.random.randn(100),
                  'b': np.random.randn(100),
                    'c': np.random.randn(100)},)
#
# h = pd.HDFStore('data.h5')
# h['obj1'] = df
# h['obj1_col1'] = df['a']
# h['obj1_col2'] = df['b']
# h['obj1_col3'] = df['c']
# print(h['obj1'])
#
# h.put('obj2',df,format='table')

#누락값 처리
a = np.array([1,2,3,4,None])
print(a)
#print(a.sum())  None으로 인해 오류발생

a = np.array([1,2,np.nan,4,5])
print(a.dtype)

print(0+ np.nan)
print(np.nan+np.nan)

print(np.nansum(a),np.nanmin(a))

pd.Series([1,2,np.nan,4,None]) # None 도 시리즈에서 np.nan으로 변경된다
s = pd.Series(range(5),dtype=int)
s[0]= None

print(s)

s = pd.Series([1,2,np.nan,'String',None])
print(s.isnull())
print(s.notnull())
s.dropna()
df.dropna(axis='columns')
df[3] = np.nan
df.dropna(axis='columns',how='all')
print(df)

print(s.fillna(0))
print(s.fillna(method='ffill')) # 앞의 값으로 nan을 채움
print(s.fillna(method='bfill')) # 뒤의 값으로 nan을 채움

print(df)
print(df.fillna(method='ffill',axis=1))

print(df.fillna(method='bfill',axis=0))
print(df.fillna(method='bfill',axis=1))

#중복제거
df = pd.DataFrame({'c1': ['a','b','c']*2 + ['b']+['c'],
                    'c2':[1,2,1,1,2,3,3,4]})
print(df)
print(df.duplicated())
print(df.drop_duplicates())

#값치환
s = pd.Series([1.,2.,-999.,-1000,4.])
print(s)
print(s.replace(-999,np.nan))  # -999는 nan으로 바꿈
print(s.replace([-999,-1000],[np.nan,0]))  # -999는 nan으로 -1000은 0으로 바꿈

















