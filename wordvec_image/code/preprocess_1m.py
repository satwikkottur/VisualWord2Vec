import os,sys

def read_csv(csv_fname):
    # Read in header information, read until [DATA]
    lines = []
    for line in open(csv_fname):
		line = line.strip()
		lines.append(line)
    return lines

def filter_sbu_with_cnn(sen, cnn):
	fid1 = open('/home/rxu2/research/data/SBU_million/SBU_captioned_photo_dataset_captions_filtered.txt','w')
	fid2 = open('/home/rxu2/research/data/SBU_million/SBU_cnn_filtered.txt','w')
	for i in range(1000):
		cnn_name = cnn + str(i+1) + '.txt'
		print cnn_name
		cdata = read_csv(cnn_name)
		clen = len(cdata)
		if clen != 1000:
			print "< 1000\n"
		for j in range(clen):
			data_line = cdata[j].split(',')
			slen = len(data_line)
			if slen == 4098:
				fid1.write('%s\n' % sen[1000*i + j])
				for k in range(1,4096):
					fid2.write("%s," % data_line[k])
				fid2.write("%s\n" % data_line[4096])
	fid1.close()
	fid2.close()			


if __name__ == "__main__":
	cnn = '/media/TOSHIBA EXT/ranxu/sbu_cnn/'
	sentances = read_csv('/home/rxu2/research/data/SBU_million/SBU_captioned_photo_dataset_captions.txt')
	filter_sbu_with_cnn(sentances, cnn)
	print "done"