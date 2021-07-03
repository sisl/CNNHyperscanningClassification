'''
Created by: Liam Kruse
Email: lkruse@stanford.edu
Modified: 06/28/2021
'''

#%%
#******************************************************************************
# SETUP
#******************************************************************************
# Imports
import keras
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Define constants
NUM_TIMESTEPS = 50; NUM_CHANNELS = 18;

# Update Matplotlib font parameters
plt.rc('font', family='serif', serif='Times New Roman')

# Define colormap
colors = ['#A7D2FC', '#7D9DBD', '#495C6F', '#1e262e']
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

#%%
#******************************************************************************
# HELPER FUNCTIONS
#******************************************************************************
""" 
read_in_sex_pred_datasets: A function to read in data of a single task type
    for dyadic sex composition prediction.
args: 
    path: Top level folder path
    task_lab: File with dyadic sex composition labels
    task_dat: File with dyadic sex composition data
    input_type: 'dtw'
returns: 
    image_data: Image data of the completed task
    label_data: Labels for mm or ff
    data_len: The number of trials
"""
def read_in_sex_pred_datasets(path, task_lab, task_dat, input_type):
    
    # Read in label data
    label_data = pd.read_csv(path + task_lab)
    label_data.drop(label_data.columns[0],axis=1,inplace=True)
    label_data = label_data.to_numpy();
    data_len = np.size(label_data)
    
    # Read in image data
    image_data = pd.read_csv(path + task_dat)
    image_data.drop(image_data.columns[0],axis=1,inplace=True)
    image_data = image_data.to_numpy();
    
    # Reshape CSV data
    if input_type == 'dtw':
        image_data = image_data.reshape(NUM_CHANNELS,NUM_CHANNELS,data_len)
    else:
        print("Bad input type")
        
    return image_data, label_data, data_len

""" 
read_in_task_pred_datasets: A function to read in data of a single sex 
    composition for task prediction.
args: 
    path: Top level folder path
    coop_lab: File with coop labels
    coop_dat: File with coop data
    comp_lab: File with comp labels
    comp_data: File with coop data
    input_type: 'dtw'
returns: 
    image_data: Image data of the tasks
    label_data: Labels for coop or comp
    data_len: The number of trials
"""
def read_in_task_pred_datasets(path, coop_lab, coop_dat, comp_lab, comp_dat, input_type):
    
    # Read in coop labels
    coop_label_data = pd.read_csv(path + coop_lab)
    coop_label_data.drop(coop_label_data.columns[0],axis=1,inplace=True)
    coop_label_data = coop_label_data.to_numpy();
    coop_len = np.size(coop_label_data)
    
    # Read in coop data
    coop_data = pd.read_csv(path + coop_dat)
    coop_data.drop(coop_data.columns[0],axis=1,inplace=True)
    coop_data = coop_data.to_numpy();

    # Reshape coop data
    if input_type == 'dtw':
        coop_data = coop_data.reshape(NUM_CHANNELS,NUM_CHANNELS,coop_len)
    else:
        print("Bad input type")
        
    # Read in comp labels
    comp_label_data = pd.read_csv(path + comp_lab)
    comp_label_data.drop(comp_label_data.columns[0],axis=1,inplace=True)
    comp_label_data = comp_label_data.to_numpy();
    comp_len = np.size(comp_label_data)
    
    # Read in comp labels
    comp_data = pd.read_csv(path + comp_dat)
    comp_data.drop(comp_data.columns[0],axis=1,inplace=True)
    comp_data = comp_data.to_numpy();
    
    # Reshape comp data
    if input_type == 'dtw':
        comp_data = comp_data.reshape(NUM_CHANNELS,NUM_CHANNELS,comp_len)
    else:
        print("Bad input type")
    
    # Concatenate data
    image_data = np.concatenate((coop_data,comp_data),axis=2)
    label_data = np.concatenate((coop_label_data,comp_label_data),axis=0)
    data_len = coop_len + comp_len
    
    return image_data, label_data, data_len

""" 
split_and_convert_data: A function to split the input data into training and
    test sets, and to convert all outputs to numpy arrays
args: 
    image_data: array of DTW similarity score data
    label_data: array of label data
    data_len: The number of trials
returns: 
    train_images: Image data for the training set
    train_labels: Labels for the training set image data
    test_images: Image data for the test set
    test_labels: Labels for the test set image data
"""
def split_and_convert_data(image_data, label_data, data_len):
    
    # Store image data in a list
    image_list = [0]*(data_len)
    for i in range(data_len):
        image_list[i] = image_data[:,:,i]
    labels = list(label_data)
    
    # Split the data into training and test sets
    train_images, test_images, train_labels, test_labels = \
        train_test_split(
            image_list, labels, test_size=0.3, random_state=16670)
     
    # Convert all data to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    return train_images, train_labels, test_images, test_labels
    
""" 
build_model: A function to build a CNN model in Keras
args: 
returns: 
    model: A Keras sequential model
"""
def build_model():
    
    # Build a sequentia model in Keras
    model = keras.Sequential([
    
        keras.layers.Conv2D(input_shape=(NUM_CHANNELS,NUM_CHANNELS,1), 
                            filters=6, kernel_size=5, strides=1, 
                            padding="same", activation=tf.nn.relu),
        # Uncomment the AveragePooling2D layers to recover a CNN architecture
        # based on the LeNet-5 design
        #keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", 
                            activation=tf.nn.relu),
        #keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)    
    ])
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    
    return model

""" 
fit_model: A function to reshape input data to the correct dimensions and fit
    the Keras model
args: 
    model: A Keras sequential model
    train_images: Image data for the training set
    train_labels: Labels for the training set image data
    test_images: Image data for the test set
    test_labels: Labels for the test set image data
returns: 
    hist: A history object for the Keras model callbacks
    reshaped_test_images: Reshaped image data for the test set
    cat_test_labels: Categorical labels for the test set 
"""
def fit_model(model, train_images, train_labels, test_images, test_labels):
    
    # Reshape the image data
    train_images = train_images.reshape(train_images.shape[0], NUM_CHANNELS, 
                                        NUM_CHANNELS, 1)
    test_images = test_images.reshape(test_images.shape[0],  NUM_CHANNELS, 
                                      NUM_CHANNELS, 1)
    
    # Convert the label data to binary class matrices
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
    hist = model.fit(train_images, train_labels, epochs=20, batch_size=32)
    
    return hist, test_images, test_labels

""" 
plot_confusion_matrix: A function to plot a confusion matrix for a given
    classification task
args: 
    c_matrix: An array of data for the confusion matrix
    title: A title for the confusion matrix
returns: 
"""
def plot_confusion_matrix(c_matrix, title):
    fig, ax = plt.subplots()
    sns.heatmap(c_matrix, annot=True, annot_kws={"size": 24}, cmap = cmap,
                fmt="d", ax = ax, cbar=False)
    plt.xlabel('Predicted', fontsize = 16)
    plt.ylabel('Actual', fontsize = 16)
    plt.title(title, fontsize = 16)
    ax.tick_params(labelsize=16)
    plt.gca().set_aspect('equal')
    
""" 
perform_classification: A function to perform a given classification task
args: 
    train_images: Image data for the training set
    train_labels: Labels for the training set image data
    test_images: Image data for the test set
    test_labels: Labels for the test set image data
    title: A title for the confusion matrix 
returns: 
"""  
def perform_classification(train_images, train_labels, test_images, test_labels, title):

    # Concatenate input data
    inputs = np.concatenate((train_images, test_images), axis=0)
    targets = np.concatenate((train_labels, test_labels), axis=0)

    # Define the K-fold Cross Validator
    num_folds = 3
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_number = 1
    
    # Initialize structures for the confusion matrix
    accuracies = np.zeros(3)
    c_matrix = np.zeros([2,2], dtype=int)
    
    # Perform classification tasks with cross-fold validation
    for train, test in kfold.split(inputs, targets):
    
        model = build_model()
        
        hist, reshaped_test_images, cat_test_labels = \
            fit_model(model, train_images, train_labels, 
                      test_images, test_labels)
        test_loss, test_acc = model.evaluate(
            reshaped_test_images, cat_test_labels, verbose=2)
        accuracies[fold_number-1] = test_acc
        
        y_pred = model.predict(reshaped_test_images)
        y_pred = np.argmax(y_pred, axis=1)
        temp_matrix = confusion_matrix(test_labels, y_pred)
        c_matrix = c_matrix + temp_matrix
        
        print('\nTest accuracy:', test_acc)
        fold_number = fold_number + 1
    
    overall_acc = np.mean(accuracies)
    print('\nOverall cooperation accuracy:', overall_acc)
    
    plot_confusion_matrix(c_matrix, title)

#%%
#******************************************************************************
# READ IN DATA
#******************************************************************************
# mm task prediction
path = "dataset/"
mm_image_data, mm_label_data, mm_data_len = \
    read_in_task_pred_datasets(path,'mm_coop_labels_dtw.csv', 
                               'mm_coop_data_dtw.csv',
                               'mm_comp_labels_dtw.csv', 
                               'mm_comp_data_dtw.csv', 'dtw')
     
#%%
# ff task prediction
path = "dataset/"
ff_image_data, ff_label_data, ff_data_len = \
    read_in_task_pred_datasets(path,'ff_coop_labels_dtw.csv', 
                               'ff_coop_data_dtw.csv',
                               'ff_comp_labels_dtw.csv', 
                               'ff_comp_data_dtw.csv', 'dtw')
#%%   
# coop sex prediction
path = "dataset/"
coop_image_data, coop_label_data, coop_data_len = \
    read_in_sex_pred_datasets(path,'coop_sex_pred_labels_dtw.csv',
                                 'coop_sex_pred_data_dtw.csv', 'dtw')

# comp sex prediction
path = "dataset/"
comp_image_data, comp_label_data, comp_data_len = \
    read_in_sex_pred_datasets(path,'comp_sex_pred_labels_dtw.csv',
                                 'comp_sex_pred_data_dtw.csv', 'dtw')     

#%%
#******************************************************************************
# SPLIT DATA INTO TRAIN AND TEST SETS
#******************************************************************************
# Male-male dyad data
mm_train_images, mm_train_labels, mm_test_images, mm_test_labels = \
    split_and_convert_data(mm_image_data, mm_label_data, mm_data_len)
    
# Female-female dyad data
ff_train_images, ff_train_labels, ff_test_images, ff_test_labels = \
    split_and_convert_data(ff_image_data, ff_label_data, ff_data_len)

# Cooperation task data
coop_train_images, coop_train_labels, coop_test_images, coop_test_labels = \
    split_and_convert_data(coop_image_data, coop_label_data, coop_data_len)
  
# Competition task data
comp_train_images, comp_train_labels, comp_test_images, comp_test_labels = \
    split_and_convert_data(comp_image_data, comp_label_data, comp_data_len)
     
#%%
#******************************************************************************
# PERFORM CLASSIFICATION
#******************************************************************************
# MM Task Classification
title = "MM Task Classification"
perform_classification(mm_train_images, mm_train_labels, 
                       mm_test_images, mm_test_labels, 
                       title)
# FF Task Classification
title = "FF Task Classification"
perform_classification(ff_train_images, ff_train_labels, 
                       ff_test_images, ff_test_labels, 
                       title)

# Coop Sex Classification
title = "Coop Sex Classification"
perform_classification(coop_train_images, coop_train_labels, 
                       coop_test_images, coop_test_labels, 
                       title)

title = "Comp Sex Prediction"
perform_classification(comp_train_images, comp_train_labels, 
                       comp_test_images, comp_test_labels, 
                       title)
