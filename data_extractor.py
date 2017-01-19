#!/usr/bin/env python
import json
import sys
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import pdb

""" The data extractor is based on timestamp categories.i.e. fileds that shared the same timestamp are extracted into same dictionary"""


def extractor(folder_name):
	data = []
	can_data = [{'label': 'str_angle', 'pid': '025', 'fields': ['timestamp','ssa'], 'data': None},
		 {'label': 'brk_press', 'pid': '224', 'fields':['timestamp','pbrk'], 'data': None},
		 {'label': 'accel_ped', 'pid': '245', 'fields':['timestamp','hv_accp'], 'data': None},
		{'label': 'speed', 'pid': '0B4', 'fields': ['timestamp','sp1'], 'data': None},
		{'label': 'parking', 'pid': '3BC', 'fields': ['timestamp','b_p'], 'data': None},
		{'label': 'yaw_rate', 'pid': '024', 'fields': ['timestamp','yr'], 'data': None},
		{'label': 'road_slope', 'pid': '320', 'fields': ['timestamp','aslp'], 'data': None},
		{'label': 'brk_state', 'pid': '3BB', 'fields': ['timestamp','b_stpe'], 'data': None},
		{'label': 'gear_pos', 'pid': '6C0', 'fields': ['timestamp','psw_pmn'], 'data': None},
		{'label': 'odo', 'pid': '611', 'fields': ['timestamp','odo'], 'data': None}
		]
	gps_data = {'label': 'gps_log', 'pid': 'gps', 'fields': ['timestamp', 'gps_timestamp', 'latitude', 'longitude', 'elevation', 'speed','climb_speed'], 'data': None}

	files = ['face_cam',  'hand_cam',  'outside_cam', 'gps_log.txt', 'lane_log.csv', 'obd_log.csv', 'CAN_log.json', 'object_log.csv']

	print 'Processing ', os.path.split(os.path.abspath(folder_name))[1]
	try:
		os.chdir(folder_name)
	except:
		print "You need to enter a subdiectory directory or full path"
		exit()
	filelist = os.listdir('.')
	if not 'CAN_log.json' in list(filelist):
		if 'CAN_log.csv' in list(filelist):
			os.system('~/can_converter-0.9.jar CAN_log.csv CAN_log.json')
			filelist.append('CAN_log.json')
		else:
			print "CAN data not found"
		
	filelist = [f for f in filelist if f in files]
		
	print "Files to be processed \n ", filelist

	json_files = [f for f in filelist if (len(f.split('.'))>1 and f.split('.')[1] in ['json', 'txt'])]

	csv_files = [f for f in filelist if (len(f.split('.'))>1 and f.split('.')[1] in ['csv'])]

	img_folders = [f for f in filelist if (len(f.split('.'))==1 and f.split('_')[1] in ['cam'])]

	#print "Parsing JSON files ..."
	json_text = []
	for file_name in json_files:
		with open(file_name, 'r') as f:
			try:
				text = json.loads(f)
			except:
				text = []
				try:
					for line in f:
						text.append(json.loads(line))
				except:
					pass
		json_text.append(text)

	if 'CAN_log.json' in json_files:
		print "Processing CAN_log ..."
		for i in range(len(can_data)):
			texts = []
			for item in json_text[json_files.index('CAN_log.json')]:
				if re.match(can_data[i]['pid'], item['pid']):
				    texts.append(item)
			texts = pd.DataFrame.from_dict(texts)
			try:
				texts = texts.loc[:,can_data[i]['fields']]
				texts['timestamp'] = pd.to_numeric(texts['timestamp'])
				texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us')
				texts.set_index('timestamp', drop=True, inplace=True)
				can_data[i].update({'data': texts})
			except:
				pass
	for ind, item in enumerate(can_data): 
		if 'ssa' in item['data'].columns:
			can_data[ind]['data']['ssa'] = can_data[ind]['data']['ssa'].apply(lambda x: x if x >= 0 else -(3070.5+x))
		#if 'sp1' in item['data'].columns:
		#	can_data[ind]['data']['sp1'] = can_data[ind]['data']['sp1']*0.621371 # convert speed from km/h to mph
	data.extend(can_data)
	del can_data
	if 'gps_log.txt' in json_files:
		print "Processing gps_log ..."
		texts = pd.DataFrame.from_dict(json_text[json_files.index('gps_log.txt')])
		texts = texts.loc[:,gps_data['fields']]
		#texts['timestamp'].apply(lambda x: int(x*1e6))
		texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='s')
		#texts['gps_timestamp'] = pd.to_datetime(texts['gps_timestamp'], unit='s')
		texts.set_index('timestamp', drop=True, inplace=True)
		texts['speed'].columns = 'gps_speed'
		gps_data.update({'data': texts})
	data.append(gps_data)
	del gps_data, json_text

	print "Processing csv based logs e,g lane, objects and obd"

	for files in csv_files:
		try:
			texts = pd.read_csv(files)
			texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us')
			texts.set_index('timestamp', drop=True, inplace=True)
			data.append({'label': files.split('.')[0], 'pid': files.split('_')[0], 'fields': texts.keys(), 'data':texts})
		except:
			pass
	print "Processing image folders ..."
	for folds in img_folders:
		try:
			img_files = glob(os.getcwd()+'/'+folds+'/*')
			texts = pd.DataFrame({folds: img_files})
			texts['timestamp'] = texts[folds].apply(lambda x:x.split('/')[-1].split('.')[0])
			#texts['timestamp'] = pd.to_numeric(texts['timestamp'])
			texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us') 
			texts = texts[['timestamp',folds]]
			texts.sort_values(by='timestamp', inplace=True)
			texts.set_index('timestamp', drop=True, inplace=True)
			data.append({'label': folds, 'pid': folds, 'fields': texts.keys(), 'data':texts})
		except:
			pdb.set_trace()
			pass
	data = [item for item in data if item['data'] is not None]
	#print [item['data'].shape for item in data] 
	return data

if __name__ == "__main__":
	args = sys.argv[1:]
	folder_name = args[0]
	data_out = extractor(folder_name)
	#pdb.set_trace()
	print "Saving extracted data to pickle file ..."
	pickle.dump(data_out, open(os.path.split(os.getcwd())[1]+'_extr.pkl','wb'))
	
	if len(args) > 1:
		print "Preparing the visualization plots ..."
		for i in range(len(data_out)):
			try:
				data_out[i]['data'].plot(kind='line')
			except:
				pass
		plt.show()

# to load do
# data = pickle.load(open(file_name,'rb'))

