## plot
import matplotlib.pyplot as plt
import seaborn as sns

## others
from glob import glob
import itertools

import numpy as np
# Scikit learn
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, resample, class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

## keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers, regularizers


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
# from tensorflow import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### Define Loss and Accuracy plot function
def loss_acc_plot(historyname, epochs_num, title, yloss_limit1=0, yloss_limit2=1.5, yacc_limit1=0.4, yacc_limit2=1):
    fig, (ax1,ax2) = plt.subplots(nrows = 2, sharex = True, figsize=(8,8));
    # plt.title(title, fontsize = 20, y=1.05)
    # Loss plot
    ax1.plot(historyname.history['loss'], 'r', label = 'Train Loss', linewidth=2)
    ax1.plot(historyname.history['val_loss'], 'b', label = 'Test Loss', linewidth=2)
    ax1.legend(loc =1)
    ax1.set_xlabel('Epochs', fontsize = 18)
    ax1.set_xticks(np.arange(0,epochs_num+1,20))
    ax1.set_ylabel('Crossentropy Loss', fontsize = 18)
    ax1.set_ylim(yloss_limit1,yloss_limit2)
    ax1.set_title('Loss Curve', fontsize = 18)
    
    # Accuracy plot
    ax2.plot(historyname.history['accuracy'], 'r', label = 'Train Accuracy', linewidth=2)
    ax2.plot(historyname.history['val_accuracy'], 'b', label = 'Test Accuracy', linewidth=2)
    ax2.legend(loc =4)
    ax2.set_xlabel('Epochs', fontsize = 18)
    ax1.set_xticks(np.arange(0,epochs_num+1,20))
    ax2.set_ylabel('Accuracy', fontsize =18)
    ax2.set_ylim(yacc_limit1,yacc_limit2)
    ax2.set_title('Accuracy Curve', fontsize =18)
    
    fig.suptitle(title, fontsize = 20, y=1.001)
    
    plt.tight_layout()
    
### Function to print out the model's loss and accuracy score
def xtest_loss_acc(modelname, X_test, y_test):
    
    model_score = modelname.evaluate(X_test, y_test, verbose =2)
    model_labels = modelname.metrics_names
    
    print(f"cnn {model_labels[0]}: {round(model_score[0] ,5)}")
    print(f"cnn {model_labels[1]}: {round(model_score[1] ,5)}")


### Define function to predict X_test, return y_pred & y_true and print the classification report
def class_report(modelname, X_test, y_test, le):
    ### predict the X_test
    # pred = modelname.predict_classes(X_test) # deprecated
    predict_x=modelname.predict(X_test) 
    pred=np.argmax(predict_x,axis=1)
    
    # compile predicted results
    y_true, y_pred = [], []
    classes = le.classes_
    
    for idx, preds in enumerate(pred):
        y_true.append(classes[np.argmax(y_test[idx])])
        y_pred.append(classes[preds])
    
    print(classification_report(y_true, y_pred,digits=4))
    return y_true, y_pred



def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()

    # plt.show()
    
def plot_confusion_matrix_sns(y_true, y_pred, classes):
    plt.figure(figsize=(10, 7))
    tick_marks = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred)
    # convert to percentage and plot the confusion matrix
    cm_pct = cm.astype(float) / cm.sum(axis =1)[:,np.newaxis]
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues', linewidths=2, linecolor='black') #cmap='Blues'
    plt.xticks(tick_marks, classes, horizontalalignment='center', rotation=70, fontsize=12)
    plt.yticks(tick_marks, classes, horizontalalignment="center", rotation=0, fontsize=12)
#     ax.set_xticklabels(
# #     ax.get_xticklabels(),
# #     # rotation=45,
# #     # horizontalalignment='right',
# #     fontsize = 2
# # )
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    

# Define fuction to build model

def cnn_model(filter_one, kernel_size_one, input_shapes,
                filter_two, kernel_size_two, dropout_two,
                filter_three, kernel_size_three, dropout_three,
                neurons, dropout_four, regularizer_rate,
                opt_learning_rate):
    
    cnn = Sequential()
    
    # add 1st convolutional layer
    cnn.add(Conv2D(filters = filter_one, kernel_size =kernel_size_one, activation = 'relu',
                   padding = 'same', input_shape = input_shapes ))
    cnn.add(MaxPooling2D())
    
    # 2nd convolutional layer
    cnn.add(Conv2D(filters =  filter_two, kernel_size = kernel_size_two, padding = 'same', activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(dropout_two))
    
    # 3rd convolutional layer
    cnn.add(Conv2D(filters =  filter_three, kernel_size = kernel_size_three, padding = 'same', activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(dropout_three))
    
    # flatten results so that it can pass to FC layer
    cnn.add(Flatten())
    
    # 1st Fully Connected (dense) layer
    cnn.add(Dense(neurons, activation = 'relu', kernel_regularizer=regularizers.l2(regularizer_rate)))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(dropout_four))
    
    # add final layer with 4 neurons(4 label)
    cnn.add(Dense(4, activation = 'softmax'))
    cnn.summary()
    
    ### Compile model
    Ad = optimizers.Adam(learning_rate=opt_learning_rate)
    cnn.compile(loss = 'categorical_crossentropy', optimizer = Ad, metrics = ['accuracy'])
    
    return cnn