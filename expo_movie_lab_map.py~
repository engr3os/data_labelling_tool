#!/usr/bin/env python
import json
import sys
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from PIL import Image
import os
from glob import glob
import numpy as np
import pdb
from os.path import expanduser
import gmplot
#import spynner
#import mechanize
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
import time

""" This script read in the pickle file and display images and make plots 
"""

sys.path.append(expanduser("~"))
from data_labeling_tool.data_sync import sync


if __name__ == "__main__":
	args = sys.argv[1:]
	file_name = args[0]
	if len(args) > 1: 
		labels = args[1:-1]
		labels_file = args[-1]
	else:
		labels_file = raw_input("Please enter the full path to labels csv file: ")
		labels = raw_input("Please enter your movie label: ")
		#labels = labels.split()
	try:
		label_data = pd.read_csv(labels_file)
		#label_data['timestamp'] = pd.to_numeric(label_data['timestamp'])#.convert_objects(convert_numeric=True)
		label_data['timestamp'] = pd.to_datetime(label_data['timestamp'], unit='ns')
		label_data.set_index('timestamp', drop=True, inplace=True)
	except:
		pass
		
plot_labels = ['ssa', 'pbrk', 'hv_accp', 'sp1', 'b_p', 'yr', 'latitude', 'longitude', 'id', 'age', 'object_class', 'sizeY', 'throttle_position','leftlane_valid', 'leftlane_confidence', 'leftlane_boundarytype', 'rightlane_valid', 'rightlane_confidence','rightlane_boundarytype', 'face_cam', 'hand_cam', 'outside_cam']
img_labels = ['face_cam', 'hand_cam', 'outside_cam']

#attr = []
#attr_count = []
#for num, label in enumerate(labels):
#	attr.append(raw_input("Please enter attributes for label "+label+" (still space separated): ").split())
#	attr_count.append(len(attr[num]))
#num_class = max(attr_count)

file_name = os.path.abspath(file_name)
if os.path.isfile(file_name):
	print "Processing", os.path.split(file_name)[1]
	try:
		data = pd.read_csv(f)
		data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
		data.set_index('timestamp', drop=True, inplace=True)
	except:
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
	os.chdir(os.path.dirname(file_name))
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted csv file"
	print "Processing", os.path.split(file_name)[1]
	data = pd.read_csv(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv')
	data['timestamp'] = pd.to_numeric(data['timestamp'])#.convert_objects(convert_numeric=True)
	data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
	data.set_index('timestamp', drop=True, inplace=True)
	os.chdir(file_name)
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted pickle file"
	print "Processing", os.path.split(file_name)[1]
	with open(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.pkl', 'rb') as f:
		data = pickle.load(f)
	os.chdir(file_name)
else:
	data = sync(file_name)
	os.chdir(file_name)
	
for item in plot_labels:
	if not item in data.columns:
		data[item] = 0

for item in img_labels:
	if not item in data.columns:
		img_labels.remove(item)
			
data_sync = data.reindex(index=label_data.index, method='ffill')
data = pd.concat([data_sync, label_data], axis=1)
try:
	start_play, stop_play = input("If you know the start and stop timestamps, \nplease enter timestamps quoted and comma separated: ")
	data = data.loc[pd.to_datetime(start_play, unit='us'):pd.to_datetime(stop_play, unit='us')+pd.to_timedelta(5, unit='s')]
	print "Start play: ", pd.to_datetime(start_play, unit='us'), "\nEnd play: ", pd.to_datetime(stop_play, unit='us')
except ValueError:
	data = data.loc[pd.to_datetime(start_play):pd.to_datetime(stop_play)+pd.to_timedelta(5, unit='s')]
	print "Start play: ", pd.to_datetime(start_play), "\n End play: ", pd.to_datetime(stop_play)
except:
	print "Invalid timestamp format, data is not changed"
	#pass
			
plt_fig = plt.figure(figsize=(20, 10), dpi=300)

## Braking
#data['pbrk'] = 5*data['pbrk']
#data.loc[data.index[(data['pred_label'] == 2) & (data['pbrk'] < 1e-6)],'pred_label'] = 0  # Not tested
data.loc[data.index[(data['pred_label'] == 1) & (data['pbrk'] > 1e-8)],'pred_label'] = 2  # Never trained
data.loc[data.index[(data['pred_label'] == 0) & (data['pbrk'] > 1e-8)],'pred_label'] = 2  # Never trained
data.loc[data.index[(data['pred_label'] == 2)],'pred_label'] = np.nan  # display nothing

pdb.set_trace()
ax5 = plt.subplot2grid((1,2), (0,1))
plt_line3, = ax5.plot([], [], 'r*', label='Predicted Braking')
plt_line6, = ax5.plot([], [], 'b-', label='Actual Brake Press')
ax5.set_ylim(min(data[['pbrk', 'pred_label']].min().min(), -0.5), min(5, 0.5+data[['pbrk', 'pred_label']].max().max()))
ax5.set_xlim(0, (data.shape[0]-1))
ax5.set_ylabel('Breaking')
plt.legend(handles=[plt_line3, plt_line6])

## Log and lat
ax7 = plt.subplot2grid((1,2), (0,0))
plt.setp(ax7.get_xaxis(), visible=False)
plt.setp(ax7.get_yaxis(), visible=False)


FFMpegWriter = an.writers['ffmpeg']
#FFMpegWriter = an.writers['imagemagick']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
#writer = FFMpegWriter(fps=30, metadata=metadata)
writer = FFMpegWriter(fps=10)

ff = webdriver.Firefox()
os.system("touch mymap.html")
map_file = "file://"+os.path.abspath("mymap.html")
gmap = gmplot.GoogleMapPlotter(data['latitude'].dropna()[0], data['longitude'].dropna()[0], 16)
#gmap.grid(data['latitude'].dropna()[0], data['latitude'].dropna()[-1], 0.001, data['longitude'].dropna()[0], data['longitude'].dropna()[-1], 0.001)
gmap.plot(data['latitude'].dropna(), data['longitude'].dropna(), 'cornflowerblue', edge_width=5)
gmap.draw("mymap.html")
#pdb.set_trace()
ff.get(map_file)
time.sleep(5)
ff.save_screenshot("map_shot.png")
gps_image = ax7.imshow(Image.open("map_shot.png"))

def updatefig(itergen):
	i = itergen
	#plt.suptitle('Processing timestamp '+str((data.index[i].value)//10**3)+', remaining '+str(data.index[-1]-data.index[i]).split()[-1]+'/'+str(data.index[-1]-data.index[0]).split()[-1]+' to complete trip', fontsize=14)
	plt_line6.set_data(np.arange(i), data.loc[data.index[:i], 'pbrk'].values)
	plt_line3.set_data(np.arange(i), data.loc[data.index[:i], 'pred_label'].values)
	ax5.set_xlim(max(0,i-450), min(i+50, data.shape[0]-1))
	if i%10 == 0:
		try:
			gmap = gmplot.GoogleMapPlotter(data['latitude'][:i].dropna()[-1], data['longitude'][:i].dropna()[-1], 20)
			gmap.plot(data['latitude'][:i].dropna(), data['longitude'][:i].dropna(), 'cornflowerblue', edge_width=3)
			gmap.draw("mymap.html")
			ff.get(map_file)
			time.sleep(5)
			ff.save_screenshot("map_shot.png")
			gps_image.set_array(Image.open("map_shot.png"))
			#os.system("rm map_shot.png")
		except:
			pass
		
	#return plt_image, plt_line, plt_line1, plt_line2, plt_line3, plt_line4, plt_line5, plt_line6, plt_line7,text, start, stop, data_label 
	return plt_line3, plt_line6, ax5, gps_image 

plt.tight_layout()
plt_ani = an.FuncAnimation(plt_fig, updatefig, frames=data.shape[0], interval = 20, blit=False, repeat=False) # blit false due to slider
#pylab.get_current_fig_manager().window.showMaximized()
plt_ani.save(labels+".mp4", writer=writer)
#plt_ani.save("data_moview.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
