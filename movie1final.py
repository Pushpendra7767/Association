# import packages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
# print datasets
df = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\movie1.csv")
df.columns
print(df)
# split dataset
df1 = df.iloc[:,0:5]
df2 = df.iloc[:,5:]
df2.columns
print(df2)
mo = df2.columns
# function for print values
mo1 = []
for i in df2.columns:
    mo1.append(df2[i].value_counts())
mo2 = []
for i in range(len(mo1)):
    mo2.append(list(mo1[i])[1])
# implement association rules & apriori algorithm
mov = apriori(df2,min_support=0.005,max_len=3,use_colnames=True)
rules = association_rules(mov,metric="lift",min_threshold=1)
rules.head()
rules.sort_values("lift",ascending=False,inplace=True)
# convert string into integer values with the help of lambda
mov["itemsets"] = mov["itemsets"].apply(lambda x: list(x)[0]).astype("unicode")
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

# plot bar graph
# itemsets
plt.bar(mov.itemsets,height=mov.support,color=['yellow'])
plt.xticks(mov["itemsets"],rotation=90)
plt.xlabel("moovie")
plt.ylabel("support")
# mo
plt.bar(mo,height=mo2,color=['red','black'])
plt.xticks(mo,rotation=90)
plt.xlabel("NAME")
plt.ylabel("COUNT")   


















  
    