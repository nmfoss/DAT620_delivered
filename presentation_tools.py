import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def FT_to_matrix(data):
    M = np.zeros((len(data), 100))
    for index in range(len(data)):
        M[index] = data[index]
        
    return M

def plot_2d(xy, Y, typ='all'):
    temp_dict = {
        'x': xy[:, 0],
        'y': xy[:, 1],
        'label': Y
    }
    
    df = pd.DataFrame.from_dict(temp_dict)

    groups = df.groupby('label')

    colors = []
    for r, g, b in np.random.uniform(0,1,size=(len(groups), 3)):
        colors.append((r,g,b))

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_color_cycle(colors)
    ax.margins(0.05)

    if typ=='all':
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)

    if typ == 'mean':
        for name, group in groups:
            ax.plot(group.x.mean(), group.y.mean(), marker='o', linestyle='', ms=5, label=name)

    if typ == 'median':
        for name, group in groups:
            ax.plot(group.x.median(), group.y.median(), marker='o', linestyle='', ms=5, label=name)
    
    ax.legend(numpoints=1, loc='upper left')

    plt.show()

def plot_cf(y, preds, title = False):

    """
    This function prints and plots the confusion matrix.
    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """

    if not title:
        title = 'Normalized confusion matrix'

    cm = confusion_matrix(y, preds)

    classes = unique_labels(y, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(18.5, 18.5)