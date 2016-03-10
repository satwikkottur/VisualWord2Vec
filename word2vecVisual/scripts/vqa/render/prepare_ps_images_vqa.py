import render_scenes_glow
import csv
import pickle
import json
import time
import os
import sys
import re
import pdb

# Author: Satwik Kottur
# Adapted from code from Ramakrishna Vedantam

#######################################################################
def prepareRenderer():
    localwebroot='/srv/share/visualw2v/data'+'/'

    #create data
    scheme=['Index','Primary','Relation','Secondary','Image','Objects','Original','Examples','workerID','assignmentID','time','comment','completion time'];
    data=list();
    opts=dict();
    opts['<jsonfile>']="";
    opts['--format']='png';
    opts['<outdir>']=os.path.join(localwebroot,'test_dir/');
    #opts['<outdir>']=os.path.join(localwebroot,'pr_image_data/');
    opts['--overwrite']=True;
    opts['--site_pngs_dir']='/home/linxiao/public_html/clipart_interface_test/abstract_scenes_v002-master/site_pngs';
    opts['--config_dir']='/home/linxiao/public_html/clipart_interface_test/abstract_scenes_v002-master/site_data';

    render = render_scenes_glow.RenderScenes(opts);

    return render;

#######################################################################
# Main function
def main():
    # Get the renderer
    print 'Obtaining renderer'
    render = prepareRenderer();

    print 'Reading tuple dictionary'
    # Load the dictionary that gives the tuple, image id and tuple id
    dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';

    with open(dataPath + 'vqa_tuples_train_dict.txt', 'r') as fileId:
        chars = '([^<>:\(\)]*)'; #characters of interest
        tuples = [re.match('<%s:%s:%s>\(%s,%s\)\n' % ((chars,)*5), line) \
                                        for line in fileId.readlines()];

    # For each of the tuple, read the json file and image file
    imgFormat = dataPath + 'tuple_images/%012d.png';
    jsonFormat = dataPath + 'iccv_json/%d.json';

    for iterId, tup in enumerate(tuples):
        # Print progress
        print 'Rendering %s / %s scene ..' % (iterId, len(tuples))

        # Read the json
        tupleId = int(tup.group(4));

        data = json.load(open(jsonFormat % tupleId, 'r'));

        # Render each scene, one by one
        render.render_one_scene({'imgName': imgFormat % tupleId, 'scene' : data},
                                data['primaryObject']['idx'], data['primaryObject']['ins'],
                                data['secondaryObject']['idx'], data['secondaryObject']['ins'])


if __name__ == '__main__':
    main();

#######################################################################
