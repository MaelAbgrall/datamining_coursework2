# libs
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import numpy as np

def disp_confusion_matrix(num_attr, predictions, actual):
    """
    displays the confusion matrix
    """
    # confusion matrix
    confusion_mat = confusion_matrix(actual, predictions)
    
    df_cm = pd.DataFrame(confusion_mat, index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                  columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.figure(figsize = (8, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='Greys')
    title = 'Confusion Matrix for ' + num_attr + ' Attributes'
    plt.title(title)
    plt.show()
    #plt.clf()
    return confusion_mat
 
def disp_accuracy_hist(num_attr, cm, predictions, actual):
    """
    displays the accuracy histogram
    """
    # accuracy histogram
    acc = []
    for i in range(len(cm)):
        total_emote = cm[i].sum()
        acc.append(cm[i][i] / total_emote)

    bar_width = .5
    positions = np.arange(7)
    plt.bar(positions, acc, bar_width)
    plt.xticks(positions + bar_width / 2, ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'))
    title = 'Histogram of Prediction Accuracy for ' + num_attr + ' Attributes'
    plt.title(title)
    plt.xlabel('Emotions')
    plt.ylabel('Accuracy')
    plt.show()
    #plt.clf()

def accuracy_plots(num_attr, predictions, actual):
    """
    displays the confusion matrix and the accuracy histogram
    """
    disp_accuracy_hist(num_attr, disp_confusion_matrix(num_attr, predictions, actual), predictions, actual)

def edit_hist_bar(bar_list, color, label):
    """
    Changes the color and label of each bar in bar_list
    """
    # to avoid duplicate labels
    bar_list[0].set_label(label) 
    bar_list[0].set_facecolor(color)
    i = 1
    for i in range(len(bar_list)):
        bar_list[i].set_facecolor(color)

def disp_acc_summary(dct):
    """
    displays a histogram that summarizes the prediction accuracies based on
    the number of attributes
    ! this function is very specific to the experiments that have been carried out
    """
    plt.figure(figsize=(8,6))
    bars = plt.bar(list(dct.keys()), tuple(list(dct.values())))
    plt.title('Histogram of Prediction Accuracy by Number of Attributes')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Accuracy')
    
    edit_hist_bar([bars[0]], 'r', 'all attributes')
    edit_hist_bar(bars[1:4], 'b', 'top overall attributes')
    edit_hist_bar(bars[4:7], 'g', 'top attributes depending on class')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()

def overall_accuracy(dct, num_attr, predictions, actual):
    """
    Calculates overall accuracy and adds it to the dictionary
    """
    total = len(predictions)
    correct_count = (predictions == actual).sum()
    print("%d out of %d" % (correct_count, total))
    accuracy = (correct_count / total)
    print("%0.03f%% correctly predicted" % (accuracy * 100))
    dct[num_attr] = accuracy
    return accuracy