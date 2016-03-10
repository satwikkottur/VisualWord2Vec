import os
import json
import shutil
import subprocess
import re
import itertools

def ffff(x):
	x["title"]="test_opt_30_3";
	return x;


def test():
	z=read_tab_deliminated_data_file(fname);
	t=create_project("sent2scene_indoor_v01","lx1990_sandbox","new_project",z[0:5],ffff);
	sync_1("new_project");
	approve("new_project",["3EA3QWIZ4IVYK89P3W653LCVDPOTIT"]);
	print(t)



#create new projects
#project name
#template name:
#questions: a list of questions(list of dict()), to be filled into a tab deliminated file
#	at current stage, make sure that the list doesn't have tabs in it.
#fmod: a function that takes in the property as a dict() and return a new dict()
def create_qualification(template_name,account,project_name,fmod=lambda x:x):
	#making sure the inputs are reasonable
	if project_name in list_projects():
		raise Exception('create_project','project name already exists')
		return None;
	if project_name=="" or project_name[0]=='.':
		raise Exception('create_project','bad project name')
		return None;
	if not template_name in list_templates():
		raise Exception('create_project','cannot find template')
		return None;
	
	project_folder=os.path.join('../data/projects/',project_name);
	template_folder=os.path.join('../data/templates/',template_name);
	
	#create project folder
	os.mkdir(project_folder);
	
	#copy configurations over
	#properties
	prop=read_json_file(os.path.join(template_folder,"task.properties.json"));
	prop=fmod(prop);
	f=open(os.path.join(project_folder,"task.properties"),"w");
	for i in prop:
		prop[i]=str(prop[i]);
	write_dict(prop,f);
	f.close();
	shutil.copy(os.path.join(template_folder,"task.question"),os.path.join(project_folder,"task.question"));
	
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	#configure account
	change_account(account);
	#submit the HITs################
	#adding a fix to a recent SSL authorization problem introduced by Java
	command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./createQualificationType.sh -question "+os.path.abspath(os.path.join(project_folder,"task.question"))+" -properties "+os.path.abspath(os.path.join(project_folder,"task.properties"));
	submit_result=subprocess.check_output(command,shell=True);
	change_account("blank");
	
	#make a project conf so that one don't need to enter account again
	conf=dict();
	conf["account"]=account;
	conf["is_qualification"]=True;
	save_json_file(conf,os.path.join(project_folder,"task.conf"));
	
	return submit_result

########################################################what worked#####################################################

#rejectwork
def reject(project_name,hits,comments=None):
	project_folder=os.path.join('../data/projects/',project_name);
	project_conf=read_json_file(os.path.join(project_folder,"task.conf"));
	account=project_conf["account"];
	
	#fill a list of dict
	#write reject file
	reject_fname=os.path.join(project_folder,"task.reject");
	if comments is not None:
		reject_schema={"assignmentIdToReject":"","assignmentIdToRejectComment":""};
		fx=lambda x:{"assignmentIdToReject":x[0],"assignmentIdToRejectComment":x[1]};
		data=[fx(x) for x in itertools.izip_longest(hits,comments)];
		write_tab_deliminated_data_file(reject_schema,data,reject_fname);
	else:
		reject_schema={"assignmentIdToReject":""};
		fx=lambda x:{"assignmentIdToReject":x};
		data=[fx(x) for x in hits];
		write_tab_deliminated_data_file(reject_schema,data,reject_fname);
	
	#configure account
	change_account(account);
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./rejectWork.sh -rejectfile "+os.path.abspath(os.path.join(project_folder,"task.reject")+" -force ");
	submit_result=subprocess.check_output(command,shell=True);
	change_account("blank");
	return submit_result


def write_tab_deliminated_data_file(schema,data,fname):
	f=open(fname,"w");
	#header
	for i in schema.keys():
		f.write(i+"\t");
	f.write("\n");
	#data
	for i in data:
		for j in schema.keys():
			f.write(i[j]+"\t")
		f.write("\n");
	f.close();


#approvework
def approve(project_name,hits,comments=None):
	project_folder=os.path.join('../data/projects/',project_name);
	project_conf=read_json_file(os.path.join(project_folder,"task.conf"));
	account=project_conf["account"];
	
	#fill a list of dict
	#write approve file
	approve_fname=os.path.join(project_folder,"task.approve");
	if comments is not None:
		approve_schema={"assignmentIdToApprove":"","assignmentIdToApproveComment":""};
		fx=lambda x:{"assignmentIdToApprove":x[0],"assignmentIdToApproveComment":x[1]};
		data=[fx(x) for x in itertools.izip_longest(hits,comments)];
		write_tab_deliminated_data_file(approve_schema,data,approve_fname);
	else:
		approve_schema={"assignmentIdToApprove":""};
		fx=lambda x:{"assignmentIdToApprove":x};
		data=[fx(x) for x in hits];
		write_tab_deliminated_data_file(approve_schema,data,approve_fname);
	
	#configure account
	change_account(account);
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./approveWork.sh -approvefile "+os.path.abspath(os.path.join(project_folder,"task.approve")+" -force ");
	submit_result=subprocess.check_output(command,shell=True);
	change_account("blank");
	return submit_result


#getresults and thats all?
def sync_1(project_name):
	project_folder=os.path.join('../data/projects/',project_name);
	project_conf=read_json_file(os.path.join(project_folder,"task.conf"));
	account=project_conf["account"];
	
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	#configure account
	change_account(account);
	if "is_qualification" in project_conf:
		#sync qualification requests
		#read id
		f=open(os.path.join(project_folder,""));
		command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./getQualificationRequests.sh -qualtypeid "+os.path.abspath(os.path.join(project_folder,"task.success"))+" -outputfile "+os.path.abspath(os.path.join(project_folder,"task.results"));
		submit_result=subprocess.check_output(command,shell=True);
	else:
		#sync the HITs################
		command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./getResults.sh -successfile "+os.path.abspath(os.path.join(project_folder,"task.success"))+" -outputfile "+os.path.abspath(os.path.join(project_folder,"task.results"));
		submit_result=subprocess.check_output(command,shell=True);
	change_account("blank");
	return submit_result;



#create new projects
#project name
#template name:
#questions: a list of questions(list of dict()), to be filled into a tab deliminated file
#	at current stage, make sure that the list doesn't have tabs in it.
#fmod: a function that takes in the property as a dict() and return a new dict()
def create_project(template_name,account,project_name,questions,fmod=lambda x:x):
	#making sure the inputs are reasonable
	if project_name in list_projects():
		raise Exception('create_project','project name already exists')
		return None;
	if project_name=="" or project_name[0]=='.':
		raise Exception('create_project','bad project name')
		return None;
	if not template_name in list_templates():
		raise Exception('create_project','cannot find template')
		return None;
	
	project_folder=os.path.join('../data/projects/',project_name);
	template_folder=os.path.join('../data/templates/',template_name);
	
	#create project folder
	os.mkdir(project_folder);
	
	#copy configurations over
	#properties
	prop=read_json_file(os.path.join(template_folder,"task.properties.json"));
	prop=fmod(prop);
	f=open(os.path.join(project_folder,"task.properties"),"w");
	for i in prop:
		prop[i]=str(prop[i]);
	write_dict(prop,f);
	f.close();
	shutil.copy(os.path.join(template_folder,"task.question"),os.path.join(project_folder,"task.question"));
	
	#create the input file
	input_schema=read_json_file(os.path.join(template_folder,"task.input.json"));
	#convert to strings; detect tabs and replace tab with
	num_replaced=0;
	for i in questions:
		for k in input_schema.keys():
			i[k]=str(i[k]);
			tmp=re.subn("\t"," ",i[k]);
			num_replaced=num_replaced+tmp[1];
			i[k]=tmp[0];
	
	#write to file
	f=open(os.path.join(project_folder,"task.input"),"w");
	#header
	for i in input_schema.keys():
		f.write(i+"\t");
	f.write("\n");
	#data
	for i in questions:
		for j in input_schema.keys():
			f.write(i[j]+"\t")
		f.write("\n");
	f.close();
	
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	#configure account
	change_account(account);
	#submit the HITs################
	command="export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./loadHITs.sh -label "+os.path.abspath(os.path.join(project_folder,"task"))+" -input "+os.path.abspath(os.path.join(project_folder,"task.input"))+" -question "+os.path.abspath(os.path.join(project_folder,"task.question"))+" -properties "+os.path.abspath(os.path.join(project_folder,"task.properties"));
	submit_result=subprocess.check_output(command,shell=True);
	change_account("blank");
	
	#make a project conf so that one don't need to enter account again
	conf=dict();
	conf["account"]=account;
	save_json_file(conf,os.path.join(project_folder,"task.conf"));
	
	return submit_result


#write dict into A:B format common in mturk
def write_dict(d,f):
	for key in d:
		f.write(key+":"+d[key]+"\n");

def chop_quotes(t):
	if t[0]=='"' and t[-1]=='"':
		return t[1:-1];
	else:
		return t;

def read_tab_deliminated_data_file(fname):
	f=open(fname,"r");
	#read header
	first_line=f.readline();
	schema=first_line.split("\t");
	schema=[re.sub("[\n\r]","",i) for i in schema];
	schema=[i for i in schema if len(i)>0];
	schema=[chop_quotes(t) for t in schema];
	data=list();
	for line in f:
		tmp=line.split("\t");
		tmp=[re.sub("[\n\r]","",i) for i in tmp];
		tmp=[i for i in tmp if len(i)>0];
		tmp=[chop_quotes(t) for t in tmp];
		data.append(dict(itertools.izip_longest(schema,tmp)));
	return schema,data;

def list_accounts():
	accounts=os.listdir('../data/accounts/');
	accounts=[i for i in accounts if not os.path.isdir(os.path.join('../data/accounts/',i))];
	return accounts

#list project names
def list_projects():
	projects=os.listdir('../data/projects/');
	projects=[i for i in projects if os.path.isdir(os.path.join('../data/projects/',i))];
	return projects

#list template names
def list_templates():
	templates=os.listdir('../data/templates/');
	templates=[i for i in templates if os.path.isdir(os.path.join('../data/templates/',i))];
	return templates

#change the account properties in MTurk command line interface: That's why you shouldn't multithread with this module
def change_account(account):
	f=open(os.path.join('../data/accounts/',account),"r");
	acc_info=json.load(f);
	f.close();
	
	#deploy to mturk.properties
	f=open(os.path.join('./aws-mturk-clt/bin','mturk.properties'),"w");
	f.write('access_key='+acc_info["access_key"]+'\n');
	f.write('secret_key='+acc_info["secret_key"]+'\n');
	f.write('service_url='+acc_info["service_url"]+'\n');
	f.write('retriable_errors=Server.ServiceUnavailable,503\n');
	f.write('retry_attempts=6\n');
	f.write('retry_delay_millis=500\n');
	f.close();
	
	return None;

#read config file for JAVA_HOME
def read_config():
	f=open('../data/config.json',"r");
	config=json.load(f);
	f.close();
	return config;

#check funds
def get_balance(account):
	#invoke commandline tool
	mturk_dir=os.path.join(os.path.dirname(os.path.realpath('./aws-mturk-clt/')),'aws-mturk-clt');
	config=read_config();
	try:
		change_account(account);
	except:
		#if change account fails
		return None;
	try:
		balance_result=subprocess.check_output("export MTURK_JVM_ARGS=-Djdk.tls.trustNameService=true;export MTURK_CMD_HOME="+mturk_dir+";export JAVA_HOME="+config["JAVA_HOME"]+";cd "+os.path.join(mturk_dir,"bin")+";./getBalance.sh",shell=True);
	except:
		#if getbalance fails
		change_account("blank");
		return None;
	#parse results
	#change it back to a default one to be safe
	change_account("blank");
	balance=re.match(r"Your account balance: \$([0-9\.]*).*$",balance_result)
	try:
		balance=float(balance.group(1));
	except:
		#if did not find any matching
		return None;
	return balance;

def read_json_file(fname):
	f=open(fname,"r");
	item=json.load(f);
	f.close();
	return item;

def save_json_file(data,fname):
	f=open(fname,"w");
	json.dump(data,f);
	f.close();
