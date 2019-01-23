# -*- coding: utf-8 -*-

import numpy as np;
import cv2;

data = dict()

def identity(arg):
	return arg;
	
def get_v_channel(img):
	return np.max(img, axis=2);
	
def mask_out_clipped_pixels(img, upper_bound=255):
	img2=img.copy();
	img2[np.tile((upper_bound<=np.max(img, axis=2))[:, :, np.newaxis], (1, 1, 3))]=0;
	return img2;
	
def do_nothing(img=None):
	return np.array([1.0, 1.0, 1.0]);

def max_rgb(img, upper_bound=255):
	return np.max(mask_out_clipped_pixels(img, upper_bound=upper_bound), axis=(0, 1)).astype("float64");
	
def gray_world(img, upper_bound=255):
	return np.mean(mask_out_clipped_pixels(img, upper_bound=upper_bound), axis=(0, 1));
	
def iwp(img, sample_size=20, samples_count=60, upper_bound=None):
	
	img2=img;
	if (upper_bound is not None):
		img2=mask_out_clipped_pixels(img, upper_bound=upper_bound);
	
	rows, cols=img.shape[0], img.shape[1];
	
	data=np.reshape(img2, (rows*cols, 3));
	
	maxima=np.zeros((samples_count, 3));
	for i in range(samples_count):
		maxima[i, :]=np.max(data[np.random.randint(low=0, high=rows*cols, size=(sample_size)), :], axis=0);
	
	return np.mean(maxima, axis=0);

def scale_channels(img, ie=np.array([1.0, 1.0, 1.0]), upper_bound=255):
	ie*=3.0/np.sum(ie);
	img2=img.astype(ie.dtype);
	
	for ch in range(3):
		img2[:, :, ch]*=ie[ch];
	
	img2[upper_bound<img2]=upper_bound;
	return img2.astype(img.dtype);

def adjust_green(ie, green=0.445):
	
	ie=ie/sum(ie);
	difference=ie[1]-green;
	rf=ie[0]/(ie[0]+ie[2]);
	bf=ie[2]/(ie[0]+ie[2]);
	ie[0]=ie[0]+difference*rf;
	ie[2]=ie[2]+difference*bf;
	ie[1]=green;
	
	return ie;

def flash(img, a=2, upper_bound=255):
	
	img2=img.astype("float64");
	v=get_v_channel(img2);
	initial_v=v.copy();
	
	gm=np.exp(np.mean(np.log(v[v>0])));
	v/=gm;
	v/=(v+a);
	
	f=v/(initial_v+1e-9);
	
	img2*=np.tile(f[:, :, np.newaxis], (1, 1, 3));
	img2*=upper_bound/np.max(img2);
	
	return img2.astype(img.dtype);


def test1(ground_truth):
	image_path = "C:/FER/projekt-master/JPG/1.jpg";
	img=cv2.imread(image_path, 6);
	
	illumination_estimation=do_nothing;
	#illumination_estimation=gray_world;
	#illumination_estimation=max_rgb;
	#illumination_estimation=iwp;
	
	green_stabilization=identity;
	#green_stabilization=adjust_green;
	
	tonemapping=identity;
	#tonemapping=flash;
	
	img2=tonemapping(scale_channels(img, green_stabilization(illumination_estimation(img))));
	
	cv2.imshow("i", img);
	cv2.imshow("i2", img2);
	cv2.waitKey();
	
def test2(ground_truth, i):
	image_path='C:/FER/Projekt/unmasked/'+i+'.png'
	img=cv2.imread(image_path, -1);
	#img-=1024;
	#img[img<0]=0;
	
	#illumination_estimation=do_nothing;
	#illumination_estimation=gray_world;
	#illumination_estimation=max_rgb;
	illumination_estimation=iwp
	
	green_stabilization=identity
	#green_stabilization=adjust_green;
	
	tonemapping=identity;
	#tonemapping=flash

	illum = illumination_estimation(img)
	img2=tonemapping(scale_channels(img, green_stabilization(ground_truth)))
	
	f=2
	
	cv2.imwrite("C:/FER/Projekt/unmasked/original/"+i+".png", img2)

def loadData(fname):
	f = open(fname, 'r')

	i = 0
	for line in f:
		vals = line.split('\t')
		nums = vals[1].split()
		data[i] = np.array((float(nums[0]), float(nums[1]), float(nums[2])))
		i+=1

	f.close()

def main():
	fname = "C:/FER/Projekt/unmasked/data.txt"
	loadData(fname)
	#test1();
	
	for i in data:
		test2(data[i], str(i))
	
if (__name__=="__main__"):
	main()
