#import pacakages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt  

#print dataset
groceries = []
with open("C:\\Users\\ACER\\Desktop\\scrap\\association\\groceries.csv") as f:
    groceries = f.read()
 #split dataset   
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
all_groceries_list = [i for item in groceries_list for i in item]
item_frequencies = Counter(all_groceries_list)
#dataset sorted 
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
frequencies = list (reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))    

#bar graph plot
plt.bar(items[1:11],height=frequencies[1:11],color=['red','green'])
plt.xticks(items[1:11],rotation=90)
plt.xlabel("items")
plt.ylabel("count")

groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9825,:]
groceries_series.columns = ["transactions"]
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
#implement apriori & association rule.
#min_supportr=0.005, max_len=3
frequent_itemsets = apriori(X,min_support=0.005,max_len=3,use_colnames = True)
#confidence=0.2
rules = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.2)
rules.head(20)
rules.sort_values('confidence',ascending=False, inplace = True)

support=rules['support']
confidence=rules['confidence']
#plot scatter graph 
plt.scatter(support,confidence,alpha=0.6)

#function for netwok graph plot
def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

draw_graph(rules,6)

# implement apriori & association rule.
# min_supportr=0.006, max_len=5
# min_confidence = 0.5
frequent_itemsets1 = apriori(X,min_support=0.006,max_len=5,use_colnames = True)
rules1 = association_rules(frequent_itemsets1,metric='confidence',min_threshold=0.5)
rules1.head(596)
rules1.sort_values('confidence',ascending=False, inplace = True)

support1=rules1['support']
confidence1=rules1['confidence']

plt.scatter(support1,confidence1,alpha=0.6,color='blue')

draw_graph(rules1,10)

#Redundant rules can be overcome by setting the appropriate value of support an confidence


























    









    








