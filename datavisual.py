import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid') # 스타일 적용

titanic = sns.load_dataset('titanic')
print(titanic.head())


print(titanic.dropna().describe())

print(titanic.mad())

print(titanic.groupby('class').count())
# sns.countplot(y = 'class',data=titanic)
# #plt.show()
#
# sns.countplot(y = 'sex',data=titanic)
# #plt.show()
# sns.countplot(y = 'alone',data=titanic)
# plt.show()

print(titanic.groupby('class')['fare'].median())
print(titanic.query('alive == "yes" '))

print(titanic.query('alive == "yes"').groupby('class').count())

print(titanic.groupby('class')['age'].describe())

print(titanic.query('alive =="yes"').groupby('class').describe())

print(titanic.groupby('sex')['age'].aggregate([min,np.median,max]))
print(titanic.query('age>30').groupby('class').median())

print(titanic.query('fare<20').groupby('class').median())

print(titanic.groupby(['class','sex'])['age'].mean().unstack())

#sns.catplot(x='sex',y ='age', hue='class',kind = 'bar',data = titanic)

#plt.show()
#sns.catplot(x='who',y ='age', hue='class',kind = 'bar',data = titanic)

print(titanic.groupby(['class','sex'])['fare'].mean().unstack())
titanic.groupby(['class','who'])['fare'].mean().unstack()

#sns.catplot(x='sex',y ='fare',hue='class', kind='bar',data=titanic)
#plt.show()

titanic.groupby(['class','sex'])['survived'].mean().unstack()
print(titanic.pivot_table('survived',index='class',columns='who'))
#sns.catplot(x='class',y ='survived',hue='sex', kind='bar',data=titanic)
#sns.catplot(x='class',y ='survived',hue='who', kind='bar',data=titanic)
#plt.show()

age = pd.cut(titanic['age'],[0,18,40,80])
print(titanic.pivot_table('survived',['sex',age],'class'))


fare = pd.qcut(titanic['fare'],3)
print(titanic.pivot_table('survived',['who',age],[fare,'class']))
print(titanic.pivot_table('survived',index='who',columns='class',margins=True))

sns.catplot(x='class',y ='survived',col='who', kind='bar',data=titanic)
plt.show()

