import render_scenes_glow
import csv
import pickle
import json
import time
import os
import sys
import re
import pdb

def mturk_strptime(time_string):
	time_string_filtered=re.sub("GMT[^ ]*","",time_string);
	return time.strptime(time_string_filtered);

def simple_table(data):
	tblf="<table border=\"1\">%s</table>";
	rowf="<tr>%s</tr>";
	itemf="<td style=\"white-space: nowrap;\">%s</td>";

	table="";
	for i in data:
		row="";
		for j in i:
			row=row+itemf%(j);
		table=table+rowf%(row);

	table=tblf%(table);
	return table;

def de_unicode(a):
	if type(a) is dict:
		b=dict();
		for i in a:
			b[de_unicode(i)]=de_unicode(a[i]);
		return b;

	elif type(a) is list:
		b=list();
		for i in range(0,len(a)):
			b.append(de_unicode(a[i]));
		return b;

	elif type(a) is unicode:
		return a.encode("utf8");

	elif a is None:
		return 'NoneType';

	else:
		return a;



#proj='sentint_test15';
proj='sentint_full4';
f=open('/srv/share/al/mturk_bot/data/projects/%s/task.results'%(proj),'r');
csv.field_size_limit(sys.maxsize)
csvreader=csv.reader(f,delimiter='\t');

results=list();
count=0;
header=dict();
for row in csvreader:
	if count==0:
		header=row;
	else:
		results.append(dict(zip(header,row)));
	count=count+1;

f.close();

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

render=render_scenes_glow.RenderScenes(opts);

#stan's rendering func asks for data['imgName'] and data['scene']
for index, i in enumerate(results):
    # printing progress
    print index, len(results)

    if i['assignmentsubmittime']!='':
        tmp=json.loads(i['Answer.hitResult']);
        fake_primary=tmp[0];
        img_name = '%s_%s.png'%(i['assignmentid'], re.sub("[^a-zA-Z0-9_]", "_", tmp[0]['relationName']))

        # Checking the arguments for the scene
        arguments = ({'imgName':'%s_%s.png'%(i['assignmentid'],
                                                        re.sub("[^a-zA-Z0-9_]","_",tmp[0]['relationName'])),'scene':tmp[0]},
                                fake_primary['primaryObject']['idx'], fake_primary['primaryObject']['ins'],
                                fake_primary['secondaryObject']['idx'], fake_primary['secondaryObject']['ins'])

        pdb.set_trace();
       # render.render_one_scene({'imgName':'%s_%s.png'%(i['assignmentid'],
       #                                                 re.sub("[^a-zA-Z0-9_]","_",tmp[0]['relationName'])),'scene':tmp[0]},
       #                         fake_primary['primaryObject']['idx'], fake_primary['primaryObject']['ins'],
       #                         fake_primary['secondaryObject']['idx'], fake_primary['secondaryObject']['ins'])
    print(index)
