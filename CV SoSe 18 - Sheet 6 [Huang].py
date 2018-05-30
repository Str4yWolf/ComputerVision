# -*- coding: utf-8 -*-
"""
Created on Fri May 04 15:49:33 2018

@author: Huang
"""
''' Aufgabe 1 '''

import numpy as np
import matplotlib.pyplot as plt
import bilderGenerator as bg

# 2.2
np.random.seed(123)
''' 
from tensorflow import set_random_seed
set_random_seed(123)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
'''

# 1.1 checked
tr_imgs = bg.zieheBilder(500)
val_imgs = bg.zieheBilder(50)



# 1.2 checked

# imgs[0] = mean
# imgs[1] = standard deviation
# imgs[2] = binary label
def plot_data(imgs):
    
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    
    x_pos = np.where(imgs[2]==1, imgs[0], -1)
    y_pos = np.where(imgs[2]==1, imgs[1], -1)
    x_neg = np.where(imgs[2]==-1, imgs[0], -1)
    y_neg = np.where(imgs[2]==-1, imgs[1], -1)

    plt.plot(x_pos, y_pos, 'bx')
    plt.plot(x_neg, y_neg, 'rx')
    
    return [x_pos, y_pos, x_neg, y_neg]
    
# plot_data(tr_imgs)




# 1.3 checked
    
# Classify linearly    
def classify_lin(imgs, w1=0.0001, w2=-0.0002, b=0.001):
    return w1*imgs[0] + w2*imgs[1] + b


tr_classified = classify_lin(tr_imgs)
val_classified = classify_lin(val_imgs)


# validate on the similarity of the input signs
def validate_sign(prediction, actual):
    prediction_sign = map(np.sign, prediction)
    return sum(map(np.equal, prediction_sign, actual)) / float(len(actual))

validation_untrained = validate_sign(val_classified, val_imgs[2])



# 1.4 checked

# train neuron and return the corresponding parameters
def train_neuron(imgs, w1=0.0001, w2=-0.0002, b=0, l_rate=0.0000005):
    
    # initialize
    tr_w1, tr_w2, tr_b = w1, w2, b
    output = np.zeros_like(imgs[0])
    
    for i in range(len(output)):
        output[i] = tr_w1*imgs[0][i] + tr_w2*imgs[1][i] + tr_b
        
        if np.sign(output[i]) != np.sign(imgs[2][i]):
            deriv_b = 2*(tr_w1*imgs[0][i] + tr_w2*imgs[1][i] + tr_b - imgs[2][i])
            deriv_w1 = deriv_b * imgs[0][i]
            deriv_w2 = deriv_b * imgs[1][i]
            
            tr_w1 -= l_rate*deriv_w1
            tr_w2 -= l_rate*deriv_w2
            tr_b -= l_rate*deriv_b
    
    return [tr_w1, tr_w2, tr_b]


# Validation score has improved significantly
n1 = train_neuron(tr_imgs, 0.0001, -0.0002, 0.001, 0.0000005)
val_classified_1 = classify_lin(val_imgs, n1[0], n1[1], n1[2])
validation_trained_1 = validate_sign(val_classified_1, val_imgs[2])



# 1.5 checked

# Iterate through training set and optionally print out results at every iteration
def classify_iterative(tr_imgs, val_imgs, repititions=1, description=False, init=False):
    
    w1 = np.random.normal(0, 0.001)
    w2 = np.random.normal(0, 0.001)
    b = 0
    
    for i in range(repititions):  
        
        if init:
            w1 = np.random.normal(0, 0.001)
            w2 = np.random.normal(0, 0.001)
            b = 0
               
        w1, w2, b = train_neuron(tr_imgs, w1, w2, b)
        
        if description or (i == (repititions-1)):
            print ""
            print "Iteration #" + str(i+1)
            print "w1=%s, w2=%s, b=%s" % (w1, w2, b)
            val_classified = classify_lin(val_imgs, w1, w2, b)
            validation = validate_sign(val_classified, val_imgs[2])
            print "Accuracy of validation: %s%%" % (validation*100)
            print ""
            
    return [w1, w2, b]

# classify_iterative(tr_imgs, val_imgs, 100, True, True)
''' Accuracy randomly fluctuates between somewhat above 50% and 100%, every time '''



# 1.6 checked

# classify_iterative(tr_imgs, val_imgs, 100, True, False)
''' Accuracy generally improves with number of iterations, yet the values fluctuate.
It appears that the fluctuation is due to the neuron overstepping local minima and therefore
readjusting itself multiple times. Even running 100 epochs multiple times yields different
results '''



# 1.7 checked - faulty

# Plot decision boundary
def plot_decision_boundary(tr_imgs=tr_imgs, val_imgs=val_imgs):
    
    # A trained neuron
    n2 = classify_iterative(tr_imgs, val_imgs, 100, False, False)

    # initialization
    means = np.array(range(2560)).astype(float) / 10
    stds = np.array(range(1280)).astype(float) / 10
    data = np.ndarray((2, 2560*1280))
    
    # all 256*128 values to iterate through in steps of 0.1
    for idx_m, m in enumerate(means):
        for idx_s, s in enumerate(stds):
           data[0][1280*idx_m + idx_s]  = m
           data[1][1280*idx_m + idx_s]  = s
    
    # find points with low error (<0.0001)
    classified = classify_lin(data, n2[0], n2[1], n2[2])
    data_points = np.where(abs(classified)< 0.0001, data, -1)
    filtered_data = np.array(([filter(lambda x: x != -1, data_points[0]), \
                               filter(lambda x: x != -1, data_points[1])]))
           
    # plot
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    plt.plot(filtered_data[0], filtered_data[1], 'r.')
    
    return filtered_data

# plot_decision_boundary()


# Plot decision boundary with data
def plot_data_classified(tr_imgs=tr_imgs, val_imgs=val_imgs):
    img_data = plot_data(tr_imgs)
    boundary_data = plot_decision_boundary(tr_imgs, val_imgs)
    
     # plot
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    plt.plot(img_data[0], img_data[1], 'bx')
    plt.plot(img_data[2], img_data[3], 'rx')
    plt.plot(boundary_data[0], boundary_data[1], 'g.')

# plot_data_classified()




''' Aufgabe 2 '''

# 2.1 checked

# Load training data
d2 = np.load('./trainingsDatenFarbe2.npz')
trImgs2 = d2['data']
trLabels2 = d2['labels']

# Load validation data
v2 = np.load('./validierungsDatenFarbe2.npz')
valImgs2 = v2['data']
valLabels2 = v2['labels']


# get mean and std (code snipped from provided solution)
def std_mean(img):
    lenImg = len(img)/3
    rMean = np.mean(img[:lenImg])
    gMean = np.mean(img[lenImg:lenImg*2])
    bMean = np.mean(img[lenImg*2:])
    rStd = np.std(img[:lenImg])
    gStd = np.std(img[lenImg:lenImg*2])
    bStd = np.std(img[lenImg*2:])
    return np.hstack((rMean, gMean, bMean, rStd, gStd, bStd)) 


# mean and std for all images
def mean_std(tr_imgs=trImgs2, val_imgs=valImgs2):
    tr_imgs_array = np.array(map(std_mean, tr_imgs)).astype(np.float32)
    val_imgs_array = np.array(map(std_mean, val_imgs)).astype(np.float32)

    return tr_imgs_array, val_imgs_array

tr_imgs_array, val_imgs_array = mean_std()



# 2.3

# recode the labels
def recode(element, initial_encoding=[1,4,8], desired_encoding=[0,1,2]):
    for i in range(len(initial_encoding)):
        if element == initial_encoding[i]:
            return desired_encoding[i]

new_tr_labels = np.array(map(recode, trLabels2))
new_val_labels = np.array(map(recode, valLabels2))

