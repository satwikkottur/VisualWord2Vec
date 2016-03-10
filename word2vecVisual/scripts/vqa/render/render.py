import urllib, cStringIO
import urlparse
import scipy
import scipy.misc
import math
import sys
sys.path.append("/srv/share/lab_venv_2.7/lib/python2.7/site-packages");
import PIL
import numpy
import PIL.Image

def clipart_render_scenes(cliparts,path_to_bgimg):
	base_url="http://vision.ece.vt.edu/clipart_scenes_v002/interface/"
	cliparts=sorted(cliparts,key=lambda x:(x["depth"],x["z"],x["order"]),reverse=True);
	im=fetch_image_from_url(urlparse.urljoin(base_url,path_to_bgimg));
	for i in cliparts:
		item_im=fetch_image_from_url(urlparse.urljoin(base_url,i["png"]));
		item_im=clipart_rescale(item_im,i["z"]);
		item_im=clipart_flip(item_im,i["flip"]);
		im=paste(im,item_im,i["x"],i["y"]);
	return im;

def fetch_image_from_url(URL):
	print(URL)
	file = cStringIO.StringIO(urllib.urlopen(URL).read())
	img = PIL.Image.open(file)
	im_arr=numpy.asarray(img)
	file.close();
	return im_arr;

def paste(im,item_im,x,y):
	x=round(x-item_im.shape[1]/2.0);
	y=round(y-item_im.shape[0]/2.0);
	rc1=im.shape;
	rc2=item_im.shape;
	cx=lambda x,y:min(max(x,0),y)
	rect1=[cx(x,rc1[1]),cx(y,rc1[0]),cx(x+rc2[1],rc1[1]),cx(y+rc2[0],rc1[0])];
	rect2=[cx(-x,rc2[1]),cx(-y,rc2[0]),cx(-x+rc1[1],rc2[1]),cx(-y+rc1[0],rc2[0])];
	im=im.astype(float)/255;
	item_im=item_im.astype(float)/255;
	print(str([x,y]));
	print(str(rc1));
	print(str(rc2));
	print(str(rect1))
	print(str(rect2))
	im[rect1[1]:rect1[3],rect1[0]:rect1[2],0]=numpy.add(numpy.multiply(im[rect1[1]:rect1[3],rect1[0]:rect1[2],0],1-item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]),numpy.multiply(item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],0],item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]));
	im[rect1[1]:rect1[3],rect1[0]:rect1[2],1]=numpy.add(numpy.multiply(im[rect1[1]:rect1[3],rect1[0]:rect1[2],1],1-item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]),numpy.multiply(item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],1],item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]));
	im[rect1[1]:rect1[3],rect1[0]:rect1[2],2]=numpy.add(numpy.multiply(im[rect1[1]:rect1[3],rect1[0]:rect1[2],2],1-item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]),numpy.multiply(item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],2],item_im[rect2[1]:rect2[3],rect2[0]:rect2[2],3]));
	im=(im*255).astype("uint8");
	return im;

def imsave(im,fname):
	im_pil=PIL.Image.fromarray(im);
	im_pil.save(fname)


def clipart_rescale(im,scale):
	im_pil=PIL.Image.fromarray(im);
	szx=int(round(im.shape[1]*pow(0.95,scale)));
	szy=int(round(im.shape[0]*pow(0.95,scale)));
	im_pil=im_pil.resize((szx,szy));
	im=numpy.asarray(im_pil);
	return im;

def clipart_flip(im,flip):
	if flip==1:
		return numpy.fliplr(im);
	else:
		return im;


