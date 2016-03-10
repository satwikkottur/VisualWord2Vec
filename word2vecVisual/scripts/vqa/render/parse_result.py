import itertools;
import re;

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def clipart_parse_mturk_string(str):
	lst=re.split("[,:$ ]*",str);
	nitems=int(lst[0]);
	nitems_per_cat=5;
	cliparts=list();
	l_item=14;
	for i in range(0,nitems*nitems_per_cat):
		item=dict();
		item["png"]=lst[i*l_item+1];
		item["i"]=int(lst[i*l_item+2]);
		item["m"]=int(lst[i*l_item+3]);
		item["object1"]=int(lst[i*l_item+4]);
		item["object2"]=int(lst[i*l_item+5]);
		item["type"]=int(lst[i*l_item+6]);
		item["attr1"]=int(lst[i*l_item+7]);
		item["attr2"]=int(lst[i*l_item+8]);
		item["x"]=float(lst[i*l_item+9]);
		item["y"]=float(lst[i*l_item+10]);
		item["z"]=float(lst[i*l_item+11]);
		item["flip"]=int(lst[i*l_item+12]);
		item["depth"]=float(lst[i*l_item+13]);
		item["order"]=float(lst[i*l_item+14]);
		if item["x"]>=-9000:
			cliparts.append(item);
	sequence_start=nitems*nitems_per_cat*l_item+1;
	SEQFORM=5;
	op_sequence=list(grouper(SEQFORM,lst[sequence_start:-1]));
	t=re.search(r"\$\*(.*)\*$",str);
	comments=t.group(1);
	return cliparts,op_sequence,comments


def clipart_cleanup_and_integrate_def(cliparts,definitions):
	#object, attributes into a list
	#type
	#fix the trash png field
	#the rest pretty much self-explanatory
	cliparts=[fix_clipart(x,definitions) for x in cliparts];
	return cliparts

def fix_clipart(item,definitions):
	#correct the png string
	item["attributes"]=[item["object1"],item["object2"],item["attr1"],item["attr2"]];
	if item["type"]==0:
		item["png"]=definitions[item["type"]][item["object1"]]+"%02d/%02d%02d.png"%(item["object2"]+1,item["attr1"]+1,item["attr2"]+1);
	else:
		item["png"]=definitions[item["type"]][item["object1"]]+"%02d.png"%(item["attr1"]+1);
	return item;

