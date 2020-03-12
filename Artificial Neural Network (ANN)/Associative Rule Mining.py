import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


store_data =pd.read_csv('store_data.csv')
a=store_data.head()
store_data = pd.read_csv('sore_data.csv',header=None)
a =store_data
b = a.shape
a=10
records = []
for i in range(0,7501):
    records.append([str(store_data.values[i,j]) for j in range(0,20)])

z = records
association_rules = apriori(records,minSupport=0.0045,min_confidence=0.22,min_lift=3,min_length=2)
association_results = list(association_rules)
a = len(association_results)
for item in association_results:
    pair =item[0]
    items =[x for x in pair]
    print("Rule:"+items[0]+"->"+items[1])
    print("Support:"+str(item[1]))
    print("Confidence:"+str(item[2][0][2]))
    print("Lift:"+str(item[2][0][3]))
    print("===============================")
