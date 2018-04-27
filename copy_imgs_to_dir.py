import pandas as pd
import shutil
import os



def create_folder(path):
	directory = os.path.dirname(path)
	if not os.path.exists(directory):
		os.makedirs(directory)

# ----------------------------------------------------------

# df = pd.read_csv('csv/validation.csv')
# img_paths = df['paths']

# folders = pd.Series(labels)
# folders = pd.unique(folders)

# for folder in folders:
# 	folder_path = ''.join(['data/fold1/validation/',folder,'/'])
# 	create_folder(folder_path)

# 	folder_path = ''.join(['data/fold1/train/',folder,'/'])
# 	create_folder(folder_path)

# ----------------------------------------------------------

# df = pd.read_csv('csv/validation.csv')
# img_paths = df['paths']
# labels = [x.split('/')[-2] for x in img_paths]
# names = [x.split('/')[-1] for x in img_paths]

# for in_path,name,label in zip(img_paths,names,labels):
# 	out_path = ''.join(['data/fold1/validation/',label,'/',name])
# 	shutil.copy(in_path,out_path)

# ----------------------------------------------------------

# df = pd.read_csv('csv/train.csv')
# img_paths = df['paths']
# labels = [x.split('/')[-2] for x in img_paths]
# names = [x.split('/')[-1] for x in img_paths]

# for in_path,name,label in zip(img_paths,names,labels):
# 	out_path = ''.join(['data/fold1/train/',label,'/',name])
# 	shutil.copy(in_path,out_path)
	
# ----------------------------------------------------------
