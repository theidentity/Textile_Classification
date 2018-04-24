from glob import glob
import pandas as pd


X = []
y = []

folders = glob('data/train/*')
for folder in folders:
	label = folder.split('/')[-1]
	image_paths = glob(folder+'/*.jpg')

	X += [path for path in image_paths]
	y += [label for x in range(len(image_paths))]

	print len(X)
	print len(y)

df = pd.DataFrame()
df['paths'] = pd.Series(X)
df['label'] = pd.Series(y)
df.to_csv('csv/all.csv',index=False)

# ----------------------

from sklearn.model_selection import train_test_split

df = pd.read_csv('csv/all.csv')
X = df['paths']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df = pd.DataFrame()
df['paths'] = pd.Series(X_train)
df['label'] = pd.Series(y_train)
print df.shape
df.to_csv('csv/train.csv',index=False)


df = pd.DataFrame()
df['paths'] = pd.Series(X_test)
df['label'] = pd.Series(y_test)
print df.shape
df.to_csv('csv/validation.csv',index=False)

# ----------------------


image_paths = glob('data/test/*.jpg')

df = pd.DataFrame()
df['paths'] = pd.Series(image_paths)
print df.shape
df.to_csv('csv/test.csv',index=False)