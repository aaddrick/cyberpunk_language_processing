import os

def dir_setup(path):
	if not os.path.isdir(path):
		os.makedirs(path)	