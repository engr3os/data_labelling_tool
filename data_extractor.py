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

""" The data extractor is based on timestamp categories.i.e. fileds that shared the same time stamp are extracted into same dictionary"""

data = [{'label': 'str_angle', 'pid': '025', 'fields': ['timestamp','ssa'], 'data': None},
	 {'label': 'brk_press', 'pid': '224', 'fields':['timestamp','pbrk'], 'data': None},
	 {'label': 'accel_ped', 'pid': '245', 'fields':['timestamp','hv_accp'], 'data': None},
	{'label': 'speed', 'pid': '0B4', 'fields': ['timestamp','sp1'], 'data': None},
	{'label': 'parking', 'pid': '3BC', 'fields': ['timestamp','b_p'], 'data': None},
	{'label': 'yaw_rate', 'pid': '024', 'fields': ['timestamp','yr'], 'data': None}
	]
gps_data = {'label': 'gps_log', 'pid': 'gps', 'fields': ['timestamp', 'gps_timestamp', 'latitude', 'longitude', 'elevation', 'speed','climb_speed'], 'data': None} 


def extractor(folder_name):
	print 'Processing ', os.path.split(os.path.abspath(folder_name))[1]
	try:
		os.chdir(folder_name)
	except:
		print "You need to enter a subdiectory directory or full path"
		exit()

	filelist = os.listdir('.')
	if 'CAN_log.csv' in filelist:
		if not 'CAN_log.json' in filelist:
			os.system('~/can_converter-0.9.jar CAN_log.csv CAN_log.json')
			filelist = os.listdir('.')
		filelist.remove('CAN_log.csv')
	if 'data_label.csv' in filelist:
		filelist.remove('data_label.csv')

	json_files = [f for f in filelist if (len(f.split('.'))>1 and f.split('.')[1] in ['json', 'txt'])]

	csv_files = [f for f in filelist if (len(f.split('.'))>1 and f.split('.')[1] in ['csv'])]

	img_folders = [f for f in filelist if (len(f.split('.'))==1 and f.split('_')[1] in ['cam'])]

	print "Parsing JSON files ..."
	json_text = []
	for file_name in json_files:
		with open(file_name, 'r') as f:
		    try:
		        text = json.loads(f)
		    except:
		        text = []
		        for line in f:
		            text.append(json.loads(line))
		json_text.append(text)

	#fig, ax = plt.subplots()
	print "Processing CAN_log ..."
	for i in range(len(data)):
		texts = []
		for item in json_text[json_files.index('CAN_log.json')]:
		    if re.match(data[i]['pid'], item['pid']):
		        texts.append(item)
		texts = pd.DataFrame.from_dict(texts)
		try:
		    texts = texts.loc[:,data[i]['fields']]
		except:
		    pass
		texts['timestamp'] = texts['timestamp'].convert_objects(convert_numeric=True)
		texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us')
		texts.set_index('timestamp', drop=True, inplace=True)
		data[i].update({'data': texts})

	print "Processing gps_log ..."
	texts = pd.DataFrame.from_dict(json_text[json_files.index('gps_log.txt')])
	texts = texts.loc[:,gps_data['fields']]
	#texts['timestamp'].apply(lambda x: int(x*1e6))
	texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='s')
	texts['gps_timestamp'] = pd.to_datetime(texts['gps_timestamp'], unit='s')
	texts.set_index('timestamp', drop=True, inplace=True)
	texts['speed'].columns = 'gps_speed'
	gps_data.update({'data': texts})
	data.append(gps_data)

	del json_text

	print "Processing csv based logs e,g lane, objects and obd"

	for files in csv_files:
		texts = pd.read_csv(files)
		texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us')
		texts.set_index('timestamp', drop=True, inplace=True)
		data.append({'label': files.split('.')[0], 'pid': files.split('_')[0], 'fields': texts.keys(), 'data':texts})

	print "Processing image folders ..."

	for folds in img_folders:
		img_files = glob(os.getcwd()+'/'+folds+'/*')
		texts = pd.DataFrame({folds+'_image': img_files})
		texts['timestamp'] = texts[folds+'_image'].apply(lambda x:x.split('/')[-1].split('.')[0])
		texts['timestamp'] = texts['timestamp'].convert_objects(convert_numeric=True)
		texts['timestamp'] = pd.to_datetime(texts['timestamp'], unit='us') 
		texts = texts[['timestamp',folds+'_image']]
		texts = texts.sort('timestamp')
		texts.set_index('timestamp', drop=True, inplace=True)
		data.append({'label': folds, 'pid': folds, 'fields': texts.keys(), 'data':texts})
	return data

if __name__ == "__main__":
	args = sys.argv[1:]
	folder_name = args[0]
	data_out = extractor(folder_name)
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

