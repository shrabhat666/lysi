# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:50:23 2020

@author: acer
"""


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
datasheet =  pd.read_csv("E:\Excelr\Association rules\my_movies.csv")
datasheet.head()
movies = datasheet.iloc[:,5:15]
movies.head()
# Building the model for support = 0.07         ################################
freq_items = apriori(movies, min_support = 0.07, use_colnames = True) 
# Storing frequencies and items in separate variables 

import matplotlib.pyplot as plt
plt.bar(x = list(range(11,21)),height = freq_items.support[1:11],color='rgmyk');plt.xticks(list(range(11,21)),freq_items.itemsets[1:11])
plt.xlabel('movies');plt.ylabel('support')  
# Collecting the inferred rules in a dataframe 
association_rules(freq_items, metric="confidence", min_threshold=0.7)
rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
print (rules)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.shape
print(rules.head()) 
###       number of rules are 230 for support = 0.07                                ########################
######################################################################################################


#Building the model for support = 0.05  and confidence = 0.7 and lift = 1.2   ######################################
frequent_itemsets = apriori(movies, min_support = 0.09, use_colnames = True) 
#Collecting the inferred rules in a dataframe 
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.shape
print(rules.head()) 
###       number of rules are 230  same as for support = 0.09  and for support = 0.07                             ########################
######################################################################################################


#Building the model for support = 0.1  and confidence = 0.7 and lift = 1        ######################################
frequent_itemsets = apriori(movies, min_support = 0.1, use_colnames = True) 
#Collecting the inferred rules in a dataframe 
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.shape
print(rules.head()) 
###       number of rules are 238 for support = 0.1, confidence = 0.7 and lift = 1                 ########################
###       number of rules reduced to 230 for when lift = 1.2 and support = 0.1, confidence = 0.7              ########################
###       number of rules reduced to 212 for when lift = 1.5 and support = 0.1, confidence = 0.7              ########################
# rules reduces when lift is increased
######################################################################################################

#Building the model for support = 0.5  and confidence = 0.7 and lift = 1 ######################################
frequent_itemsets = apriori(movies, min_support = 0.3, use_colnames = True) 
#Collecting the inferred rules in a dataframe 
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.shape
print(rules.head()) 
###       number of rules are 238 for support = 0.1, confidence = 0.7 and lift = 1                 ########################
## number of rules reduced to 12 for when support = 0.3, confidence = 0.7   and lift = 1           ########################
## number of rules reduced to 4 for when support = 0.5, confidence = 0.7   and lift = 1           ########################
# rules reduces when support is increased                               #########################
######################################################################################################

####################### sorting books   ########################################################
from collections import Counter
all_movies_list = []
all_movies_list = [i for item in movies for i in item]
from collections import Counter
item_frequencies = Counter(all_movies_list)
len(item_frequencies)
# after sorting
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
len(item_frequencies)
#Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))
# barplot of top 10 
import matplotlib.pyplot as plt
plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("movies")
plt.ylabel("Count")

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
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


###################################################################################################
###
##################          visualization of association  rules           ##############################
###
###################################################################################################
##         Scatterplot   for support = 0.05 and confidence = 0.7 and lift = 1.2     ##############
###
###################################################################################################
print (rules)

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
import numpy as np
fit = np.polyfit(lift, confidence, 1)
fit_fn = np.poly1d(fit)
plt.plot(lift, confidence, 'yo', lift, fit_fn(lift))
###             defining a function for   draw graph                            #################

 
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
            color_map.append('pink')
       else:
            color_map.append('blue')       
 
 
   
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
