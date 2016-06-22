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
import atexit
from os.path import expanduser

""" This script read in the pickle file and display images and make plots 
"""

sys.path.append(expanduser("~"))
from data_labeling_tool.data_sync import sync

def onClick(event):
    global pause
    pause ^= True
def sp_onClick(event):
	global delta
	delta += 2
	delta = max(delta,DELTA_MIN)
def sl_onClick(event):
	global delta
	delta -= 2
	#delta = min(delta,DELTA_MIN)
def r_label(label):
	global cid, radio1
	cid = label
	try:
		label_index = labels.index(label)
	except ValueError:
		pass
	[label.set_text(attr[label_index][num]) if num < len(attr[label_index]) else label.set_text('NA') for num,label in enumerate(radio1.labels)]
	return cid
def r1_label(label):
	global clabel
	clabel = label
	return clabel
def log_label(event):
	print 'clabel: ', clabel, 'cid: ', cid
	if cid != 'NA':
		global data_label
		start_time = start.val
		stop_time = stop.val
		data_label.loc[pd.to_datetime(start_time, unit='us'):pd.to_datetime(stop_time, unit='us'),cid] = clabel
		print data_label.loc[pd.to_datetime(start_time, unit='us'):pd.to_datetime(stop_time, unit='us')]
		#dataframe = data_label[:pd.to_datetime(stop_time, unit='us')]
def save_label(event):
	print "Saving labels to file .. data_label.csv"
	dataframe = data_label.copy()
	dataframe.index = dataframe.index.astype(np.int64)//10**3
	dataframe.to_csv('data_label.csv')
	del dataframe	
def set_label_start(event):
	global start
	if int(drange.val) == 0:
		start.set_val(stop.val)

def combine_imgs(img_files, rot_index):
	imgs = [Image.open(img_file) for img_file in list(img_files) if type(img_file) in [str]]
	if rot_index < len(imgs):
		imgs[rot_index] = imgs[rot_index].rotate(180)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
	return Image.fromarray(imgs_comb)

if __name__ == "__main__":
	args = sys.argv[1:]
	file_name = args[0]
	if len(args) > 1: 
		labels = args[1:]
	else:
		labels = input("Please enter your labels, space separated with quote: ")
		labels = labels.split()
		
attr = []
attr_count = []
for num, label in enumerate(labels):
	attr.append(input("Please enter attributes for label "+label+" (still space separated): ").split())
	attr_count.append(len(attr[num]))
num_class = max(attr_count)

file_name = os.path.abspath(file_name)
if os.path.isfile(file_name):
	print "Processing", os.path.split(file_name)[1]
	try:
		data = pd.read_csv(f)
		data['timestamp'] = pd.to_datetime(data['timestamp'])
		data.set_index('timestamp', drop=True, inplace=True)
	except:
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
	os.chdir(os.path.dirname(file_name))
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted csv file"
	print "Processing", os.path.split(file_name)[1]
	with open(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv', 'rb') as f:
		data = pd.read_csv(f)
		data['timestamp'] = data['timestamp'].convert_objects(convert_numeric=True)
		data['timestamp'] = pd.to_datetime(data['timestamp'])
		data.set_index('timestamp', drop=True, inplace=True)
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted pickle file"
	print "Processing", os.path.split(file_name)[1]
	with open(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.pkl', 'rb') as f:
		data = pickle.load(f)
else:
	data = sync(file_name)
	
data_label = pd.DataFrame(columns=labels, index=data.index)
labels.insert(0, "NA")
attr.insert(0,["NA"]*num_class)
pause = False
log = False
cid = labels[0]
clabel = attr[0][0]
delta = 0
DELTA_MIN = 0
start_time = data_label.index[0]
stop_time = data_label.index[0]
img_files = data.iloc[0,-3:]
rot_index = list(data.keys()[-3:]).index('face_cam_image')
plt_fig = plt.figure()
# Texts
ax = plt.subplot2grid((3,3), (1,2))
plt.setp(ax.get_xaxis(), visible=False)
plt.setp(ax.get_yaxis(), visible=False)
text = ax.text(0.05, 0.05, 'Detected object:', transform=ax.transAxes)
# Images
#ax1 = plt_fig.add_subplot(3,1,1)
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax1.set_anchor('W')
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax1.get_xaxis(), visible=False)
plt.setp(ax1.get_yaxis(), visible=False)
plt_image = ax1.imshow(combine_imgs(img_files, rot_index))
button = Button(ax1, '')
button.on_clicked(onClick)
# steering angle
#ax2 = plt_fig.add_subplot(3,3,4)
ax2 = plt.subplot2grid((3,3), (1,0))
plt_line, = ax2.plot([], [], 'r')
ax2.set_ylim(min(data['ssa'].min(), 0), data['ssa'].max())
ax2.set_xlim(0, data.shape[0]-1)
ax2.set_ylabel('Steering angle')
# speed
ax3 = plt.subplot2grid((3,3), (1,1))
#plt_line1, = ax3.plot([], [], 'b', ms=1)
plt_line1, = ax3.plot([], [], 'b')
ax3.set_ylim(min(data['sp1'].min(), 0), data['sp1'].max())
ax3.set_xlim(0, data.shape[0]-1)
ax3.set_ylabel('Speed')
## Yaw rate
ax5 = plt.subplot2grid((3,3), (2,0))
plt_line3, = ax5.plot([], [], 'k')
ax5.set_ylim(min(data['yr'].min(), 0), data['yr'].max())
ax5.set_xlim(0, data.shape[0]-1)
ax5.set_ylabel('Yaw Rate')
## Throttle Position
ax6 = plt.subplot2grid((3,3), (2,1))
plt_line4, = ax6.plot([], [], 'r', label='from OBD(norm)')
plt_line5, = ax6.plot([], [], 'b', label='from CAN(norm)')
plt_line2, = ax6.plot([], [], 'g', label= 'Parking')
throt_diff = data['throttle_position'].max()-data['throttle_position'].min()
throt_min = data['throttle_position'].min()
accel_diff = data['hv_accp'].max()-data['hv_accp'].min()
accel_min = data['hv_accp'].min()
ax6.set_ylim(0, 1.0)
ax6.set_xlim(0, data.shape[0]-1)
ax6.set_ylabel('Throttle Position')
plt.legend(handles=[plt_line4, plt_line5, plt_line2])
## Brake Pressure
ax7 = plt.subplot2grid((3,3), (2,2))
plt_line6, = ax7.plot([], [], 'r')
ax7.set_ylim(min(data['pbrk'].min(), 0), data['pbrk'].max())
ax7.set_xlim(0, data.shape[0]-1)
ax7.set_ylabel('Brake Pressure')
ax8 = plt.axes([0.47+0.2, 0.86, 0.05, 0.04])
spbutton = Button(ax8, 'FFW >>', color='lightblue', hovercolor='0.975')
spbutton.on_clicked(sp_onClick) 
ax9 = plt.axes([0.47+0.2, 0.81, 0.05, 0.04])
slbutton = Button(ax9, 'RWD <<', color='magenta', hovercolor='0.975')
slbutton.on_clicked(sl_onClick)
ax10 = plt.axes([0.47+0.2, 0.76, 0.05, 0.04])
readybutton = Button(ax10, 'Set label \nstart time', color='yellow', hovercolor='0.975')
readybutton.on_clicked(set_label_start)
ax11 = plt.axes([0.47+0.2, 0.71, 0.05, 0.04])
logbutton = Button(ax11, 'Log label', color='green', hovercolor='0.975')
logbutton.on_clicked(log_label)
ax12 = plt.axes([0.87, 0.81, 0.05, 0.09])
savebutton = Button(ax12, 'Save labels\nto file', color='cyan', hovercolor='0.975')
savebutton.on_clicked(save_label)
startax = plt.axes([0.15, 0.625, 0.25, 0.03])
stopax = plt.axes([0.55, 0.625, 0.25, 0.03])
start = Slider(startax, 'Label Start \nTimestamp', data.index[0].value//10**3, data.index[-1].value//10**3, valinit=data.index[0].value//10**3)
stop = Slider(stopax, 'Label Stop\nTimestamp', data.index[0].value//10**3, data.index[-1].value//10**3, valinit=data.index[0].value//10**3)
start.valfmt = '%i'
stop.valfmt = '%i'
rangeax = plt.axes([0.47+0.2, 0.67, 0.18, 0.03])
drange = Slider(rangeax,'Constant\ntimedelta(s)', -100, 100, valinit=0)
drange.valfmt = '%i'
rax = plt.axes([0.53+0.2, 0.71, 0.06, 0.19])
radio = RadioButtons(rax, labels, active=0)
#radio = CheckButtons(rax, labels[1:], [False]*len(labels[1:]))
rax1 = plt.axes([0.60+0.2, 0.71, 0.06, 0.19])
radio1 = RadioButtons(rax1, ['NA']*num_class, active=0)
#radio1 = CheckButtons(rax1, range(num_class), [False]*num_class)
radio.on_clicked(r_label)
radio1.on_clicked(r1_label)
def itergen():
	i_max = data.shape[0]
	i = 0
	while i+delta < i_max:
		if not pause:
			i+=delta
		yield i
def updatefig(itergen):
	i = itergen
	stop.set_val(data_label.index[i].value//10**3)
	if int(drange.val) != 0:
		start.set_val(data_label.index[i].value//10**3-int(drange.val)*10**6)
	global img_files
	img_files = data.iloc[i, -3:]
	plt.suptitle('Processing '+str((data.index[i].value)//10**3)+' timestamp, remaining '+str(data.index[-1]-data.index[i]).split()[-1]+'/'+str(data.index[-1]-data.index[0]).split()[-1]+' to complete trip', fontsize=14)
	plt_image.set_array(combine_imgs(img_files, rot_index))
	plt_line.set_data(range(i), data.loc[data.index[:i], 'ssa'].values)
	plt_line1.set_data(range(i), data.loc[data.index[:i], 'sp1'].values)
	plt_line2.set_data(range(i), data.loc[data.index[:i], 'b_p'].values)
	plt_line3.set_data(range(i), data.loc[data.index[:i], 'yr'].values)
	plt_line4.set_data(range(i), (data.loc[data.index[:i], 'throttle_position'].values-throt_min)/throt_diff)
	plt_line5.set_data(range(i), (data.loc[data.index[:i], 'hv_accp'].values-accel_min)/accel_diff)
	plt_line6.set_data(range(i), data.loc[data.index[:i], 'pbrk'].values)
	text.set_text('Detected object: '+str(data.loc[data.index[i], 'object_class'])+'\n'+
	'Object id: '+str(data.loc[data.index[i], 'id'])+'\n'+
	'Object size: '+str(data.loc[data.index[i], 'sizeY'])+'\n'+
	'Object age: '+str(data.loc[data.index[i], 'age'])+'\n'+
	'Leftlane_valid: '+str(data.loc[data.index[i], 'leftlane_valid'])+'\n'+
	'Rightlane_valid: '+str(data.loc[data.index[i], 'rightlane_valid'])+'\n'+
	'Leftlane_confidence: '+str(data.loc[data.index[i], 'leftlane_confidence'])+'\n'+
	'Rightlane_confidence: '+str(data.loc[data.index[i], 'rightlane_confidence'])+'\n'+
	'Leftlane_boundarytype: '+str(data.loc[data.index[i], 'leftlane_boundarytype'])+'\n'+
	'Rightlane_boundarytype: '+str(data.loc[data.index[i], 'rightlane_boundarytype'])+'\n'+
	'Replay Speed: '+str(delta)+'X\n'+
	'Label timedelta: '+str(pd.to_timedelta(stop.val-start.val, unit='us')).split()[-1]+'\n'
	) 
		
	return plt_image, plt_line, plt_line1, plt_line2, plt_line3, plt_line4, text, start, stop, data_label 

#plt_fig.canvas.mpl_connect('button_press_event', onClick)
plt_ani = an.FuncAnimation(plt_fig, updatefig, itergen, interval = 10, blit=False, repeat=False)
plt.show()
