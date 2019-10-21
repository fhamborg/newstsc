import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def create_save_plotted_confusion_matrices(multilabel_confusion_matrices, expected_labels, basefilename):
    # based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    assert len(expected_labels) == multilabel_confusion_matrices.shape[0]

    for index, label_class in enumerate(expected_labels):
        binary_expected_labels = [label_class, "not_" + label_class]
        confusion_matrix = multilabel_confusion_matrices[index]
        ax, title = plot_confusion_matrix(confusion_matrix, binary_expected_labels, normalize=False,
                                          title='Label={}'.format(label_class))

        plt.savefig('statistics/{}_{}.png'.format(basefilename, title))


def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax, title


if __name__ == '__main__':
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    multiconfmat = multilabel_confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    create_save_plotted_confusion_matrices(multiconfmat, ["ant", "bird", "cat"])
