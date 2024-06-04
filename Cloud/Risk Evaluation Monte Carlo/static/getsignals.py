#!/usr/bin/python3
import sys
import json
import pandas as pd
import math
import random
import csv

inputs=sys.stdin.read()
input_extract=json.loads(inputs)

data=input_extract['Data']
d=input_extract['d']
h=input_extract['h']
t=input_extract['t']

if t=='buy':
	res=buySimResults(data,s,h)
elif t=='sell':
	res=sellSimResults(data,s,h)
	




def buySimResults(data,shots,history):
    var95list=[]
    var99list=[]
    closing=[]
    tradedate=[]
    data=pd.DataFrame(data)
    return_data_dict={}
    shots=int(shots)
    history=int(history)
    for i in range(history, len(data)):
        if data.Buy[i]==1:
            mean=data.Close[i-history:i].pct_change(1).mean()
            std=data.Close[i-history:i].pct_change(1).std()
            simulated = [random.gauss(mean,std) for x in range(shots)]
            simulated.sort(reverse=True)
            var95 = simulated[int(len(simulated)*0.95)]
            var99 = simulated[int(len(simulated)*0.99)]
            var95list.append(var95)
            var99list.append(var99)
            closing.append(data.Close[i])
            tradedate.append(data.Date[i])
            
    return_data_dict['var95']=var95list
    return_data_dict['var99']=var99list
    return_data_dict['Close']=closing
    return_data_dict['Date']=tradedate
   
    return return_data_dict
    
    
    
def sellSimResults(data,shots,history):
    var95list=[]
    var99list=[]
    closing=[]
    tradedate=[]
    data=pd.DataFrame(data)
    
    return_data_dict={}
    shots=int(shots)
    history=int(history)
    for i in range(history, len(data)):
        if data.Sell[i]==1:
            mean=data.Close[i-history:i].pct_change(1).mean()
            std=data.Close[i-history:i].pct_change(1).std()
            simulated = [random.gauss(mean,std) for x in range(shots)]
            simulated.sort(reverse=True)
            var95 = simulated[int(len(simulated)*0.95)]
            var99 = simulated[int(len(simulated)*0.99)]
            var95list.append(var95)
            var99list.append(var99)
            closing.append(data.Close[i])
            tradedate.append(data.Date[i])
            
    return_data_dict['var95']=var95list
    return_data_dict['var99']=var99list
    return_data_dict['Close']=closing
    return_data_dict['Date']=tradedate
   
    return return_data_dict








print("Content-Type:text/html;charset=utf-8")
print("")
print(str(res))


