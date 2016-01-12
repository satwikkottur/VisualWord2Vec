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
import os

# Add the path for importing src module
import sys
sys.path.append('visual_genome_python_driver/');
from src import api as vg

# Get dataset along with annotations
def getDataset(imageData, savePath, reverseInd, workerId):
    imgSaveFolder = savePath + '%d/%d/'
    imgSavePath = imgSaveFolder + '%d_%d.png';

    # For each image, get subregion and captions
    curIter = 0;
    for i in imageData:
        curIter += 1;
        print 'Saving %d / %d (%d) image...' % (curIter, len(imageData), workerId);

        # Check the folders and their existance
        imgIndex = reverseInd[i['id']];
        innerFolder = imgIndex % 1000;
        outerFolder = int(imgIndex / 1000);

        if not os.path.exists(savePath + str(outerFolder)):
            os.makedirs(savePath + str(outerFolder));

        destination = imgSaveFolder % (outerFolder, innerFolder);
        if not os.path.exists(destination):
            os.makedirs(destination);
        imgSavePath = destination + '%d_%d.png';

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

# Save captions for the image regions
def saveCaptions(imageData, capPath):
    capId = open(capPath, 'w');
    curIter = 0;
    # Save captiosn for each of the image
    for i in imageData:
        print 'Saving %d (%d) / %d image...' % (curIter, i['id'], len(imageData));
        curIter += 1;

        regions = vg.GetRegionDescriptionsOfImage(id=i['id']);
        for j in xrange(len(regions)):
            reg = regions[j];
            # Write the regions description
            regDesc = reg.phrase.encode('utf-8').strip('\n').replace('\n', ' ');
            capId.write('%d : %d : %s\n' % (i['id'], j, regDesc));

    capId.close();

# Save captions for the image regions, using multiple workers
def saveCaptionsMulti(imageData, capPath, noWorkers):
    workerId = 0;
    # List of all the current jobs
    jobs = [];

    for piece in chunks(imageData, len(imageData)/noWorkers + 1):
        # Start the thread
        thread = multiprocessing.Process(target = saveCaptions, \
                           args = (piece, '%s_%03d' % (capPath, workerId)));
        jobs.append(thread);
        thread.start();
        workerId += 1;

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

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
def saveImageList(imageData, reverseInd, listPath, savePath):
    fileId = open(listPath, 'w');

    imgSaveFolder = savePath + '%d/%d/'
    imgSavePath = imgSaveFolder + '%d_%d.png';

    # For each image, get subregion and captions
    curIter = 0;
    for i in imageData:
        curIter += 1;
        print 'Saving %d / %d image...' % (curIter, len(imageData));

        # Check the folders and their existance
        imgIndex = reverseInd[i['id']];
        innerFolder = imgIndex % 1000;
        outerFolder = int(imgIndex / 1000);

        destination = imgSaveFolder % (outerFolder, innerFolder);
        if os.path.exists(destination):
            listing = os.listdir(destination);

            # Printing the image list
            for img in listing:
                regionId = int(img.split('_')[1].split('.')[0]);
                fileId.write('%04d%04d%04d\t%s\n' % (outerFolder, innerFolder, \
                                            regionId, destination + img));
    fileId.close();

# Use the driver to get the dataset
if __name__ == '__main__':
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';
    imagePath = dataPath + 'image_data.json';
    savePath = dataPath + 'images/';
    splitCapPath = dataPath + 'captionSplits/' + 'genome_train_captions.txt';
    capPath = dataPath + 'genome_train_captions.txt';

    # First read all the images
    with open(imagePath, 'r') as fileId:
        imageData = json.load(fileId);

    # Write a list of images vs image id
    reverseInd = {};
    with open(savePath + 'index.txt', 'w') as fileId:
        for i in xrange(len(imageData)):
            fileId.write('%d : %d\n' % (i, imageData[i]['id']));
            reverseInd[imageData[i]['id']] = i;

    # Save the images, in parallel
    #jobs = [];
    #noThreads = 128;
    #for i in xrange(0, noThreads):
    #    thread = multiprocessing.Process(target=getDataset, \
    #           args=(imageData[i::noThreads], savePath, reverseInd, i));
    #    jobs.append(thread);
    #    thread.start();

    #getDataset(imageData, savePath);

    # Save the captions
    #saveCaptions(imageData, capPath);

    # Save the captions in multiple files
    saveCaptionsMulti(imageData, splitCapPath, 15);

    # Save the image lists
    #listPath = dataPath + 'image_list.txt';
    #saveImageList(imageData, reverseInd, listPath, savePath);
