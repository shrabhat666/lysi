# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:48:15 2020

@author: Shraddha Bhat
"""
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
groceries = []
# As the file is in transaction data we will be reading data directly 
with open("E:\\Excelr\\Association rules\\groceries (3).csv") as f:
    groceries = f.read()



# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
all_groceries_list = []

#for i in groceries_list:
#    all_groceries_list = all_groceries_list+i
all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter

item_frequencies = Counter(all_groceries_list)
# after sorting
#item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 

import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame for the transactions data 
import pandas as pd
# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
frequent_itemsets = apriori(X, min_support=0.005, use_colnames = True)
frequent_itemsets.shape
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
print (rules)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules.shape
rules.head(20)
rules.sort_values('confidence',ascending = False,inplace=True)
########################## for increasing confidence value the number of rules is reducing
#rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.09)
#rules.shape
#rules.head(20)
#rules.sort_values('confidence',ascending = False,inplace=True)
###  for confidence = 0.07, number of rules are 2001
###  for confidence = 0.09, number of rules are 1749
###  for confidence = 0.5, number of rules are 99
###  for confidence = 0.6, number of rules are 13
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))
ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]
rules_no_redudancy
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('confidence',ascending=False).head(10)
### selecting top 10 rules based on confidence #############################



###############################################################################################
#################               for support = 0.05                   ##########################
###############################################################################################
frequent_itemsets = apriori(X, min_support=0.05, max_len=3,use_colnames = True)
frequent_itemsets.shape
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.03)
rules.shape
rules.head(20)
rules.sort_values('confidence',ascending = False,inplace=True)
 ###  for confidence = 0.03, number of rules are 6
 ###  for confidence = 0.05, number of rules are 6
 ###  for confidence = 0.07, number of rules are 6
 ###  for confidence = 0.09, number of rules are 6

########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))
ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('confidence',ascending=False).head(10)
##########
 #         antecedents   consequents  ...  leverage  conviction
#4            (yogurt)  (whole milk)  ...  0.020379    1.244132
#0  (other vegetables)  (whole milk)  ...  0.025394    1.214013
#3        (rolls/buns)  (whole milk)  ...  0.009636    1.075696

#[3 rows x 9 columns]
###################################################################################################
###
##################          visualization of association  rules           ##############################
###
###################################################################################################
##         Scatterplot   for support = 0.005 and confidence = 0.7 and lift = 1.2     ##############
###
###################################################################################################
frequent_itemsets = apriori(X, min_support=0.005, use_colnames = True)
frequent_itemsets.shape
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
print (rules)
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
###########                    support vs confidence                         #######################
support=rules.support 
confidence=rules.confidence
lift = rules.lift
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()
###########                    support vs lift                            #######################
plt.scatter(support, lift, alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()
###########                   Lift vs Confidence                          #######################
fit = np.polyfit(lift, confidence, 1)
fit_fn = np.poly1d(fit)
plt.plot(lift, confidence, 'yo', lift, fit_fn(lift))

###             defining a function for   draw graph                            #################
import numpy as np
 
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
 #####                  draw graph  function ends                       ##########################
import seaborn as sns1
sns1.regplot(x=support, y=confidence, fit_reg=False)
plt.gcf().clear()
draw_graph(rules, 10) 


