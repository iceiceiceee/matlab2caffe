import re


project_path = "G:/py/tf2caffe/genmodel/"
model_path = project_path + 'alex.prototxt'
data_path = "G:/py/tf2caffe/modeldata/"
fp = open("G:/py/tf2caffe/genmodel/alex2.prototxt", 'a')

with open(model_path, 'r') as f:
	for line in f:
		if re.search(r'^  #[a-z][1-9]$', line) or re.search(r'^  #[a-z][1-9]_[a-z]$', line):
			with open(data_path + line[3:-1] + '.prototxt', 'r') as fd:
				fp.writelines(fd.read())
			if re.match(r'  #b[1-9]_v', line):
				fp.writelines('blobs {\n  data: 1.0\n  shape {\n    dim: 1\n  }\n}')
		if re.search(r'^  #[a-z][1-9][a-z]$', line):
			with open(data_path + line[3:-1] + '.prototxt', 'r') as fd:
				fp.writelines(fd.read())
		else:
			fp.writelines(line)
fp.close()

