#!/usr/bin/env python
# coding: utf-8

# In[98]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

train = pd.read_csv('D:/houseprices/train.csv')
train
qn = [f for f in train.columns if train.dtypes[f] != 'object'] #количественные столбцы
ql = [f for f in train.columns if train.dtypes[f] == 'object'] #качественные столбцы
qn.remove('SalePrice') 
qn.remove('Id')

missing = train.isnull().sum() #подсчёт и вывод всех столбцов с нулевыми значениями
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

y = train['SalePrice'] #построение распределения для цен и годов постройки
sns.displot(y, kde='True')
z = train['GarageYrBlt']
sns.displot(z, kde='True', bins=60)

test_norm = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01 #проверка, являются ли остальные - нормальным распределением
normal = pd.DataFrame(train[qn])
normal = normal.apply(test_norm)
print(not normal.any())


# In[99]:


f = pd.melt(train, value_vars=qn) #превращает начальный массив в список, разбивая на переменные и их значения
g = sns.FacetGrid(data=f, col="variable", col_wrap=3, sharex=False, sharey=False)
g = g.map_dataframe(sns.histplot, "value", kde='True') #построение кучи графиков


# In[116]:


for c in ql:
    train[c] = train[c].astype('category')
    if train[c].isnull().any():
        train[c] = train[c].cat.add_categories(['MISSING'])
        train[c] = train[c].fillna('MISSING')
        
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation = 90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=ql)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False, height=6)
g = g.map(boxplot, 'value', 'SalePrice')
print(f)


# Анализируя распределение по категориям и соотнося с ценой, можно сделать некоторое выводы, исходя из этих графиков.

# In[ ]:




