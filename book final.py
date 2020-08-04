# import packages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt 
# print dataset
df = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\association\\book.csv")
df.columns
print(df)
bk1=df.columns
# print values
bks=[]
for i in df.columns:
    bks.append(df[i].value_counts())
bk2=[]
for i in range(len(bks)):
    bk2.append(list(bks[i])[1])  
    
 

# implement apriori algorithm & association rules
bo = apriori(df,min_support=0.005,max_len=3,use_colnames=True)
rules = association_rules(bo,metric="lift",min_threshold=1)
rules.head(11)
rules.sort_values("lift",ascending=False, inplace=True)
#  convert string into unique integer values
bo["itemsets"] = bo["itemsets"].apply(lambda x: list(x)[0]).astype("unicode")
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

# plot bar graph 
# bk1
plt.bar(bk1,height=bk2,color=['yellow','green'])
plt.xticks(bk1,rotation=90)
plt.xlabel("name")
plt.ylabel("count") 
# itemsets
plt.bar(bo.itemsets,height=bo.support,color=['red'])
plt.xticks(bo["itemsets"],rotation=90)
plt.xlabel('books')
plt.ylabel('support')

    
    