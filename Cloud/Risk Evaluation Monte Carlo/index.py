import os 
import logging 
import math
import random
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pandas_datareader import data as pdr
import multiprocessing
import time
from flask import jsonify
#from flask_googlecharts import GoogleCharts

import http.client
import json
import ast

from flask import Flask, request, render_template,session
 
app = Flask(__name__)
#charts = GoogleCharts(app) multiprocessing
 
# various Flask explanations available at:  
# https://flask.palletsprojects.com/en/1.1.x/quickstart/ 


'''
def doRender(tname, values={}):
	
	if not os.path.isfile( os.path.join(os.getcwd(), 'templates/'+tname) ): #No such file
		return render_template('index.htm')
	return render_template(tname, **values)
'''
##Helper Functions
# Load data in s3 bucket
def load_bucket():
	import http.client
	import json
	
	data=loadData()
	
	data['Buy']=0
	data['Sell']=0
	
	sig_data=getBuySellSignals(data)
	
	datetemp=sig_data.reset_index()
	
	datetemp['Date'] = pd.to_datetime(datetemp['Date']).dt.date
	datetemp['Date']=datetemp['Date'].astype(str)
	
	d_sig_data=datetemp.to_dict('records')
	
	request_dict={'Data':d_sig_data}
	jdata=json.dumps(request_dict)
	# sending the data to s3 bucket to go ahead with analyse
	try:
		c = http.client.HTTPSConnection("fdp2d1oi5j.execute-api.us-east-1.amazonaws.com")
		json= '{ "Key1": ' + jdata + '}'
		c.request("POST", "/default/data_load", json)
		response = c.getresponse()
		res = response.read().decode('utf-8')
		
		return True #terminal check
		
	except IOError:
		print( 'Failed to Load Data ' , host )
#Load Data		
def loadData():
	yf.pdr_override()
	today = date.today()
	decadeAgo = today - timedelta(days=1095)

	data = pdr.get_data_yahoo('BP.L', start=decadeAgo, end=today) 
	return data
		
#get signaled data	
def getBuySellSignals(data):
	for i in range(2, len(data)): 
		body = 0.01 
	 # Three Soldiers 
		if (data.Close[i] - data.Open[i]) >= body  and data.Close[i] > data.Close[i-1]  and (data.Close[i-1] - data.Open[i-1]) >= body  and data.Close[i-1] > data.Close[i-2]  and (data.Close[i-2] - data.Open[i-2]) >= body: data.at[data.index[i], 'Buy'] = 1 
		#print("Buy at ", data.index[i]) 
		
	 # Three Crows 
		if (data.Open[i] - data.Close[i]) >= body and data.Close[i] < data.Close[i-1] and (data.Open[i-1] - data.Close[i-1]) >= body and data.Close[i-1] < data.Close[i-2] and (data.Open[i-2] - data.Close[i-2]) >= body: data.at[data.index[i], 'Sell'] = 1 
		#print("Sell at ", data.index[i]) 
	return	data
	

def parallel_getsigvar9599():
	try:
		c = http.client.HTTPSConnection("4n0wrlb158.execute-api.us-east-1.amazonaws.com")
		c.request("GET", "/default/get_sig_vars9599")
		response = c.getresponse()
		data = response.read().decode('utf-8')
		data=json.loads(data)
		
		sorted_data=pd.DataFrame(data)
		
		sorted_data['Date'] = pd.to_datetime(sorted_data['Date'])
		sorted_data.sort_values(by='Date', ascending = False, inplace = True)
		lim_data=sorted_data.head(20)
		lim_data = lim_data.iloc[:, :2]
		lim_data=lim_data.to_dict('list')
		return lim_data
		
	except IOError:
		print( 'Failed to open ' , host ) # Is the Lambda address correct?

def parallel_get_avg_var9599():
	try:
		c = http.client.HTTPSConnection("2fk1nh3c9f.execute-api.us-east-1.amazonaws.com")
		c.request("GET", "/default/get_avg_vars9599")
		response = c.getresponse()
		data = response.read().decode('utf-8')
		return data
		
	except IOError:
		print( 'Failed to open ' , host ) # Is the Lambda address correct?

def parallel_get_sig_profit_loss():
	try:
		c = http.client.HTTPSConnection("ih2vq89s74.execute-api.us-east-1.amazonaws.com")
		c.request("GET", "/default/get_sig_profit_loss")
		response = c.getresponse()
		data = response.read().decode('utf-8')
		
		temp_dict=json.loads(data)
		
		
		return temp_dict[:20]
		
	except IOError:
		print( 'Failed to open ' , host ) # Is the Lambda address correct?

def parallel_get_chart_url():
	try:
		c = http.client.HTTPSConnection("hychwp0vwc.execute-api.us-east-1.amazonaws.com")
		c.request("GET", "/default/get_chart_url")
		response = c.getresponse()
		data = response.read().decode('utf-8')
		return data
		
	except IOError:
		print( 'Failed to open ' , host ) # Is the Lambda address correct?
def parallel_get_tot_profit_loss():
	try:
		c = http.client.HTTPSConnection("e1wlp4ss45.execute-api.us-east-1.amazonaws.com")
		c.request("GET", "/default/get_tot_profit_loss")
		response = c.getresponse()
		data = response.read().decode('utf-8')
		return data
		
	except IOError:
		print( 'Failed to open ' , host ) # Is the Lambda address correct?

	



		 
# Warmup
@app.route('/warmup',methods=['POST'])
def Warmup():
	import http.client
	import json
	import ast
	global start_time,s_val,r_val,first_analyse
	
	first_analyse=False
	start_time=time.perf_counter()
	s=request.get_json('s')
	s_val=s.get('s')
	r_val=s.get('r')
	
	session['s']=s.get('s')
	print(start_time)
	
	jsondata=json.dumps(s)
	
	print(jsondata)
	if request.method == 'POST':
        	
        	c = http.client.HTTPSConnection("gz6o3121v7.execute-api.us-east-1.amazonaws.com")
        	json= '{ "Key1": ' + jsondata + '}'
        	#json= '{ "key1": "'+jsondata+'"}'
        	c.request("POST", "/default/warmup", json)
        	response = c.getresponse()
        	data = response.read().decode('utf-8')
        	#jsdata=ast.literal_eval(data)
        	global end_time
        	end_time=time.time()
        	
        	
        	return data


# Resources Ready
@app.route('/resources_ready',methods=['GET'])
def resourcesReady():
	import http.client
	import json
	
	
	host = "ut29zr93xa.execute-api.us-east-1.amazonaws.com"
	
	try:
		c = http.client.HTTPSConnection(host)
		h = {'Content-type': 'application/json'}
		
		c.request("GET", "/default/resources_ready")
		response= c.getresponse()
		data = response.read().decode('utf-8')
		
		return data
		
	
	except IOError:
        	print( 'Failed to open ' , host ) # Is the Lambda address correct?
	
	
	
	
	
	
	
	
	
# Get Warmup Cost
@app.route('/get_warmup_costs',methods=['GET'])
def getWarmupCost():
	
	#Cost = (Number of invocations per month) * (Execution time per invocation in seconds) * (Memory allocated in GB) * (Cost per GB-seconds)
	#Cost = 1,000,000 * (300ms/1000ms) * (256MB/1024) * $0.00001667 Cost = $13.33
	
	
	timeconsumed=round(end_time-start_time,2)
	r=int(r_val)
	s=s_val
	if s=='lambda':
			cost=float(timeconsumed)*1000*0.0000000167*float(r)
	elif s=='ec2':
			cost=float(timeconsumed)/3600*0.0116*float(r)		
	
	return {"billable_time":timeconsumed,"cost":cost}
	
	
	return 	return_dict	

# Get Endpoints
@app.route('/get_endpoints',methods=['GET'])
def getEndpoints():
	end_points={}
	if s_val == 'lambda': 
		end_points['resource_1'] = "https://gz6o3121v7.execute-api.us-east-1.amazonaws.com/default/warmup"
		end_points['resource_2'] = "https://ut29zr93xa.execute-api.us-east-1.amazonaws.com/default/resources_ready"
		end_points['resource_3'] = "https://jjlphgefo3.execute-api.us-east-1.amazonaws.com/default/analyse"
		end_points['resource_4'] = "https://4n0wrlb158.execute-api.us-east-1.amazonaws.com/default/get_sig_vars9599"
		end_points['resource_5'] = "https://e1wlp4ss45.execute-api.us-east-1.amazonaws.com/default/get_tot_profit_loss"
		end_points['resource_6'] = "https://ih2vq89s74.execute-api.us-east-1.amazonaws.com/default/get_sig_profit_loss"
		end_points['resource_7'] = "https://2fk1nh3c9f.execute-api.us-east-1.amazonaws.com/default/get_avg_vars9599"
		end_points['resource_8'] = "https://hychwp0vwc.execute-api.us-east-1.amazonaws.com/default/get_chart_url"
		end_points['resource_9'] = "https://of7b2k54el.execute-api.us-east-1.amazonaws.com/default/terminate"
	else:
		end_points['resource'] = ""
		
	
	return end_points


# Resources Ready
@app.route('/terminate')
def Terminate():
	import http.client
	import json
	import ast
	
	host = "of7b2k54el.execute-api.us-east-1.amazonaws.com"
	
	try:
		c = http.client.HTTPSConnection(host)
		h = {'Content-type': 'application/json'}
		c.request("GET", "/default/terminate")
		response= c.getresponse()
		data = response.read().decode('utf-8')
		
		return data
	except IOError:
        	return {''}
	
	
# Analyse Function
@app.route('/analyse',methods=['POST'])
def GetAnalyse():
	import http.client
	import json
	import concurrent
	import concurrent.futures
	import time
	
		
	check=load_bucket()
	global first_analyse
	first_analyse=True
	print(first_analyse)
	global get_sig_var9599_time_consumed
	global get_avg_var9599_time_consumed
	global get_sig_profit_loss_time_consumed
	global get_tot_profit_loss_time_consumed
	global get_chart_time_consumed
	
	global h,d,t,p
	
	h=request.get_json().get('h')
	d=request.get_json().get('d')
	t=request.get_json().get('t')
	p=request.get_json().get('p')
	
	get_sig_var9599_time_consumed=0
	get_avg_var9599_time_consumed=0
	get_sig_profit_loss_time_consumed=0
	get_tot_profit_loss_time_consumed=0
	get_chart_time_consumed=0
	try:
		s=request.get_json('s')
		jsondata=json.dumps(s)
		
		
		c = http.client.HTTPSConnection("jjlphgefo3.execute-api.us-east-1.amazonaws.com")
		json= '{ "Key1": ' + jsondata + '}'
		c.request("POST", "/default/analyse", json)
		response = c.getresponse()
		data = response.read().decode('utf-8')
		
		return data
	except IOError:
		return{''}

@app.route('/get_sig_vars9599')
def GetSigVar9599():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent
	import concurrent.futures
	import time
	
	global get_sig_var9599_time_consumed
	
	if first_analyse:
		
		try:
			resources = int(r_val)
		except:
			print( 'Please Wawrmup Resources first')
		
		try:
			start = time.perf_counter()
			with concurrent.futures.ThreadPoolExecutor(max_workers=resources) as executor:
				futures = {executor.submit(parallel_getsigvar9599): i for i in range(resources)}
				return_dict = {}
				for future in concurrent.futures.as_completed(futures):
					index = futures[future]
					result = future.result()
					return_dict[index] = result
			end = time.perf_counter()
			get_sig_var9599_time_consumed+=round(end-start,2)
			print("Elapsed Time:", end - start)
			
			return return_dict #separate values can obtained by indexing
		except IOError:
			return{''}
	else:
		return {'':''}
# Get Avg Var 95 99
@app.route('/get_avg_vars9599')
def GetAvgVar9599():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	global get_avg_var9599_time_consumed
	if first_analyse:
		
		try:
			resources = int(r_val)
		except:
			print( 'Please Wawrmup Resources first')
		
		try:
			start = time.perf_counter()
			with concurrent.futures.ThreadPoolExecutor(max_workers=resources) as executor:
				futures = {executor.submit(parallel_get_avg_var9599): i for i in range(resources)}
				return_dict = {}
				for future in concurrent.futures.as_completed(futures):
					index = futures[future]
					result = future.result()
					return_dict[index] = result
				
			var95avg_total = 0
			var99avg_total = 0
			num_entries = 0
			print(return_dict)
			for key, value in return_dict.items():
				entry = json.loads(value)
				var95avg_total=var95avg_total+entry["var95avg"]
				var99avg_total=var99avg_total+entry["var99avg"]
				num_entries += 1
			var95avg_avg = var95avg_total / num_entries
			var99avg_avg = var99avg_total / num_entries
			print("Average var95avg:", var95avg_avg)
			print("Average var99avg:", var99avg_avg)
			end = time.perf_counter()
			get_avg_var9599_time_consumed+=round(end-start,2)
			return {'var95':var95avg_avg,'var99':var99avg_avg}
			
		except IOError:
			return {''}
	else:
		return {'':''}

#Get Sig Profit
@app.route('/get_sig_profit_loss')
def GetSigProfitLoss():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	global get_sig_profit_loss_time_consumed
	if first_analyse:
		try:
			resources = int(r_val)
		except:
			print( 'Please Wawrmup Resources first')
		
		try:
			start = time.perf_counter()
			with concurrent.futures.ThreadPoolExecutor(max_workers=resources) as executor:
				futures = {executor.submit(parallel_get_sig_profit_loss): i for i in range(resources)}
				return_dict = {}
				for future in concurrent.futures.as_completed(futures):
					index = futures[future]
					result = future.result()
					return_dict[index] = result
				
			
			end = time.perf_counter()
			get_sig_profit_loss_time_consumed+=round(end-start,2)
			#return {'var95':var95avg_avg,'var99':var99avg_avg}
			return return_dict
			
		except IOError:
			return {''}
	else:
		return {'':''}

#Get Tot Sig Profit
@app.route('/get_tot_profit_loss')
def GetTotSigProfitLoss():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	global get_tot_profit_loss_time_consumed
	
	if first_analyse:
		try:
			resources = int(r_val)
		except:
			print( 'Please Wawrmup Resources first')
		
		try:
			start = time.perf_counter()
			with concurrent.futures.ThreadPoolExecutor(max_workers=resources) as executor:
				futures = {executor.submit(parallel_get_tot_profit_loss): i for i in range(resources)}
				return_dict = {}
				for future in concurrent.futures.as_completed(futures):
					index = futures[future]
					result = future.result()
					return_dict[index] = result
				
			
			end = time.perf_counter()
			get_tot_profit_loss_time_consumed+=round(end-start,2)
			#return {'var95':var95avg_avg,'var99':var99avg_avg}
			return return_dict
			
		except IOError:
			return {''}
	else:
		return {'':''}


#Get Chart  URL
@app.route('/get_chart_url')
def GetChartUrl():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	global get_chart_time_consumed
	if first_analyse:
		try:
			resources = int(r_val)
		except:
			print( 'Please Wawrmup Resources first')

		try:
			start = time.perf_counter()
			with concurrent.futures.ThreadPoolExecutor(max_workers=resources) as executor:
				futures = {executor.submit(parallel_get_chart_url): i for i in range(resources)}
				return_dict = {}
				for future in concurrent.futures.as_completed(futures):
					index = futures[future]
					result = future.result()
					return_dict[index] = result
				
			
			end = time.perf_counter()
			get_chart_time_consumed+=round(end-start,2)
			#return {'var95':var95avg_avg,'var99':var99avg_avg}
			return return_dict
			
		except IOError:
			return {''}
	else:
		return {'':''}

#Get Time Cost  URL
@app.route('/get_time_cost')
def GetTimeCost():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	if first_analyse:
		total_time_consumed=get_sig_var9599_time_consumed+get_avg_var9599_time_consumed+get_sig_profit_loss_time_consumed+get_tot_profit_loss_time_consumed+get_chart_time_consumed
		print(total_time_consumed)
		return_dict={}
		
		resources = s_val
		r=r_val
		try:
			if resources=='lambda':
				cost=float(total_time_consumed)*1000*0.0000000167*float(r) #used 1024 memory
			elif resources=='ec2':
				cost=float(total_time_consumed)/3600*0.0134*float(r)		
			return {"billable_time":total_time_consumed,"cost":cost}
			
		except IOError:
			return{''}
	else:
		return {'':''}



# Audit
@app.route('/get_audit')
def GetAudit():
	#lambda config - eph:2048,Memory:1028,timeout:10
	import http.client
	import json
	import concurrent.futures
	import time
	
	response_dict={}
	
	res_data_varAvg=GetAvgVar9599()
	res_data_profitLoss=GetTotSigProfitLoss()
	res_data_timeCost=GetTimeCost()
	
	
	print(res_data_profitLoss)
	print("###################################")
	print(res_data_timeCost)
	print("###################################")
	print(res_data_varAvg)
	
	response_dict['s']=s_val
	response_dict['r']=r_val
	response_dict['h']=h
	response_dict['d']=d
	response_dict['t']=t
	response_dict['p']=p
	
	for value in res_data_profitLoss.values():
		value_dict=eval(value)
		response_dict['profit_loss']=value_dict['profit_loss'][0]
		
	
	
	response_dict['av95']=res_data_varAvg['var95']
	response_dict['av99']=res_data_varAvg['var99']
	
	response_dict['time']=res_data_timeCost['billable_time']
	response_dict['cost']=res_data_timeCost['cost']
	
	
	
	try:	
		
		jsondata=json.dumps(response_dict)
		c = http.client.HTTPSConnection("93p8ji60i0.execute-api.us-east-1.amazonaws.com")
		json= '{ "Key1": ' + jsondata + '}'
		c.request("POST", "/default/audit", json)
		response = c.getresponse()
		data = response.read().decode('utf-8')
		
			
		return data
	
	except IOError:
        	return {''}
	
	
	
@app.route('/reset')
def GetReset():
	global get_sig_var9599_time_consumed
	global get_avg_var9599_time_consumed
	global get_sig_profit_loss_time_consumed
	global get_tot_profit_loss_time_consumed
	global get_chart_time_consumed
	global first_analyse
	
	get_sig_var9599_time_consumed=0
	get_avg_var9599_time_consumed=0
	get_sig_profit_loss_time_consumed=0
	get_tot_profit_loss_time_consumed=0
	get_chart_time_consumed=0
	first_analyse=False
	
	return {'result':'Ok'}
		
@app.route('/resources_terminated')
def GetResourcesTerminated():
	import http.client
	import json
	
	
	host = "y2wpxflo33.execute-api.us-east-1.amazonaws.com"
	
	try:
		c = http.client.HTTPSConnection(host)
		h = {'Content-type': 'application/json'}
		
		c.request("GET", "/default/resources_terminated")
		response= c.getresponse()
		data = response.read().decode('utf-8')
		
		return data
		
	
	except IOError:
        	print( 'Failed to open ' , host ) # Is the Lambda address correct?
	
	



	


#############################################
# for ec2
@app.route('/cacheavoid/<name>')
def cacheavoid(name):
    # file exists?
    if not os.path.isfile( os.path.join(os.getcwd(), 'static/'+name) ): 
        return ( 'No such file ' + os.path.join(os.getcwd(), 'static/'+name) )
    f = open ( os.path.join(os.getcwd(), 'static/'+name) )
    contents = f.read()
    f.close()
    return contents # far from the best HTTP way to do this



	
#############################################other functs##########################################################
		
 # catch all other page requests - doRender checks if a page is available (shows it) or not (index)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def mainPage(path):
	return doRender(path)

 
@app.errorhandler(500) 
# A small bit of error handling 
def server_error(e): 
    logging.exception('ERROR!') 
    return """ 
    An  error occurred: <pre>{}</pre> 
    """.format(e), 500 
 
if __name__ == '__main__': 
    
    # Entry point for running on the local machine 
    # On GAE, endpoints (e.g. /) would be called. 
    # Called as: gunicorn -b :$PORT index:app, 
    # host is localhost; port is 8080; this file is index (.py) 
    app.secret_key = 'super secret key'
    app.run(host='127.0.0.1', port=7080, debug=True)

