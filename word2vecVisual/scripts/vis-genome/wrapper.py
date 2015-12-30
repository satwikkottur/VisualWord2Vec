# Utilities for visual-genome dataset
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
import requests
from cStringIO import StringIO
import urllib
import pdb
import multiprocessing
import sys

# Add the path for importing src module
import sys
sys.path.append('visual_genome_python_driver/');
from src import api as vg

# Get dataset along with annotations
def getDataset(imageData, savePath, workerId):
    imgSavePath = savePath + '%d_%d.png';
    # For each image, get subregion and captions
    curIter = 0;
    for i in imageData:
        curIter += 1;
        print 'Saving %d / %d (%d) image...' % (curIter, len(imageData), workerId);

        fileURL = StringIO(urllib.urlopen(i['url']).read());
        img = PIL_Image.open(fileURL);
        regions = vg.GetRegionDescriptionsOfImage(id=i['id']);
        maxDims = img.size;
        for j in xrange(len(regions)):
            reg = regions[j];

            # Ensure the dimensions are properly cropped
            cropDims = [reg.x, reg.y, reg.x + reg.width, reg.y + reg.height];
            cropDims[0] = max(0, cropDims[0]);
            cropDims[1] = max(0, cropDims[1]);
            cropDims[2] = min(maxDims[0], cropDims[2]);
            cropDims[3] = min(maxDims[1], cropDims[3]);

            # Do not save invalid crops
            if (cropDims[0] >= cropDims[2]) or (cropDims[1] >= cropDims[3]):
                continue;

            # Save the cropped file
            try:
                img.crop(tuple(cropDims)).save(imgSavePath % (i['id'], j));
            except:
                print cropDims, maxDims, reg, workerId
                sys.exit(0);

def saveCaptions(imageData, capPath):
    capId = open(capPath, 'w');
    # Save captiosn for each of the image
    for i in imageData:
        print 'Saving %d / %d image...' % (i['id'], len(imageData));

        regions = vg.GetRegionDescriptionsOfImage(id=i['id']);
        for j in xrange(len(regions)):
            reg = regions[j];
            # Write the caption
            capId.write('%d : %d : %s\n' % (i['id'], j, reg.phrase.encode('utf-8')));

    capId.close();

# Visualizating the regions
def visualize_regions(image, regions):
    response = requests.get(image.url);
    img = PIL_Image.open(StringIO(response.content));
    plt.imshow(img);
    ax = plt.gca();
    for region in regions:
        ax.add_patch(Rectangle((region.x, region.y),
                                region.width,
                                region.height,
                                fill=False,
                                edgecolor='red',
                                linewidth=3));
        ax.text(region.x, region.y, region.phrase, \
            style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10});
    fig = plt.gcf();
    plt.tick_params(labelbottom='off', labelleft='off');
    plt.show();
    fig = plt.gcf();
    fig.set_size_inches(18.5, 10.5);

# Create the list of images for extracting vgg features
## Getting the list of images
##basePath = 'srv/share/vqa/release_data/abstract_v002/scene_img/img/%d.png';
#basePath = '/home/satwik/vqa_images/%d.png';
#
## Write the image list
#with open('/home/satwik/imageList.txt', 'w') as fileId:
#    [fileId.write('%d\t%s\n' %(i, basePath % i)) for i in xrange(0, 50000)];
#
# Use the driver to get the dataset
if __name__ == '__main__':
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';
    imagePath = dataPath + 'image_data.json';
    savePath = dataPath + 'images/';
    capPath = dataPath + 'genome_train_captions.txt';

    # First read all the images
    with open(imagePath, 'r') as fileId:
        imageData = json.load(fileId);

    # Save the images, in parallel
    #jobs = [];
    #noThreads = 48;
    #for i in xrange(0, noThreads):
    #    thread = multiprocessing.Process(target=getDataset, \
    #                args=(imageData[i::noThreads], savePath, i));
    #    jobs.append(thread);
    #    thread.start();

    #getDataset(imageData, savePath);

    # Save the captions
    saveCaptions(imageData, capPath);
