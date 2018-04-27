import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import numpy as np
import itertools

size = 20
y_pred = np.random.randint(low=0, high=2, size=size)
y_actual = np.random.randint(low=0, high=2, size=size)
y_actual[:size/2] = y_pred[:size/2]


def key_stats_binary(y_pred,y_actual):
	print 'Accuracy : ',accuracy_score(y_actual,y_pred)
	print 'Precision : ',precision_score(y_actual,y_pred)
	print 'Recall : ',recall_score(y_actual,y_pred)
	print 'F1 : ',f1_score(y_actual,y_pred)

	roc = roc_curve(y_actual,y_pred)
	print roc
	x,y = roc
	plt.plot(x,y)
	plt.show()

def key_stats(y_pred,y_actual):
	print 'Accuracy : ',accuracy_score(y_actual,y_pred)
	print 'Accuracy : ',f1_score(y_actual,y_pred)
	print 'Accuracy : ',accuracy_score(y_actual,y_pred)

def plot_confusion_matrix(y_pred,y_actual,target_names,normalize=False):

    cm = confusion_matrix(y_actual, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.savefig('tmp/conf_matrix.png')


# plot_confusion_matrix(y_pred,
# 	y_actual,
# 	target_names=['item'+str(i) for i in range(18)],
# 	normalize=False)

key_stats_binary(y_pred,y_actual)