# Author: Mark Harmon
# Purpose: Make a few figures based upon the sequence to sequence models
# Plot the maximum probabilities as k increases.  Sequences of 5-10. How often does the model make the exact
# same prediction? How many neutral preditictions does my model make

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
from scipy import stats
import scipy.spatial.distance

def kullback_leibler_divergence(y_true,y_pred):
    y_true = np.clip(y_true, 1e-07, 1)
    y_pred = np.clip(y_pred, 1e-07, 1)
    return np.sum(y_true * np.log(y_true/y_pred), axis=-1)

def total_variation_distance(y_true,y_pred):
    y_true = np.clip(y_true, 1e-07, 1)
    y_pred = np.clip(y_pred, 1e-07, 1)
    return np.linalg.norm(y_true-y_pred,ord=np.inf,axis=-1)
def crossentropy(y_true,y_pred):
    y_true = np.clip(y_true, 1e-07, 1)
    y_pred = np.clip(y_pred, 1e-07, 1)
    return -np.sum(y_true*np.log(y_pred),axis=-1)
def oldest_main():

    main_add = '/home/mharmon/FinanceProject/ModelResults/tickseq30'

    # Get out all of the 5 predictions first and calculate the number of neutral predictions vs. non-neutral
    # Also how often is it the exact same predictions?

    sub_add = main_add + 'win5/Predictions/'
    pred_list = os.listdir(sub_add)

    # Predictions shape is (examples,stocks,time,class)
    same_pred = 0
    neutral_pred = 0
    total_pred = 0
    for i in range(len(pred_list)):
        load_add = sub_add + pred_list[i]
        pred,dates = pkl.load(open(load_add,'rb'))
        pred = np.argmax(pred,axis=3)

        for k in range(len(pred)):

            for s in range(5):

                pred_mode = stats.mode(pred[k,s,:])[0][0]
                num_same = len(np.where(pred[k,s,:]==pred_mode)[0])

                neutral_where = np.where(pred[k,s,:]==2)[0]
                if len(neutral_where)>0:
                    neutral_pred+= len(neutral_where)

                same_pred += num_same
                total_pred +=5

    neutral_avg = neutral_pred/float(total_pred)
    same_avg = same_pred/float(total_pred)


    print('%.4f Neutral Predictions' %neutral_avg)
    print('%.4f Same Predictions' %same_avg)

    # Now we have to make figure for maximum probability
    pngcount = 0

    for i in range(6):
        window = i+5
        sub_add = main_add + 'win' + str(window) + '/Predictions/'

        pred_list = os.listdir(sub_add)

        decreasing=0
        total=0
        distance_max = 0
        distance_bottom=0
        max_counter = 0
        max_bottom=0
        for j in range(len(pred_list)):
            load_add = sub_add + pred_list[j]
            pred,dates = pkl.load(open(load_add,'rb'))


            for k in range(len(pred)):

                for s in range(5):

                    if np.max(pred[k,s,0,:])>np.max(pred[k,s,-1,:]):
                        distance_max+=1
                    distance_bottom+=1

                    # No I need to compute average value.
                    mean_len = int(np.ceil(window/2.))
                    beg_maxes = np.mean(np.max(pred[k,s,0:mean_len,:]),axis=1)
                    end_maxes = np.mean(np.max(pred[k,s,mean_len:,:]),axis=1)

                    if beg_maxes>end_maxes:
                        max_counter+=1
                    max_bottom+=1

                    for w in range(window-1):

                        first_max = np.max(pred[k,s,w,:])
                        next_max = np.max(pred[k,s,w+1,:])

                        if next_max<first_max:
                            decreasing+=1

                        total+=1

        prec_dec = 100*(decreasing/float(total))
        distance_dec = distance_max/float(distance_bottom)
        avg_max_value = max_counter/float(max_bottom)
        print('Percentage of time we have decreasing probability for window %r is %.4f' %(window,prec_dec) )
        print('Decrease from beginnging to end for window %r is %.4f' %(window,distance_dec))
        print('Average percentage for decrease for window %r is %.4f' %(window,avg_max_value))
    return

def old_main():

    main_add = '/home/mharmon/FinanceProject/ModelResults/tickseq30'

    # Get out all of the 5 predictions first and calculate the number of neutral predictions vs. non-neutral
    # Also how often is it the exact same predictions?

    sub_add = main_add + 'win5/Predictions/'
    pred_list = os.listdir(sub_add)

    # The goal here is to look at the average max values
    overall_larger = 0
    under_larger =0
    window = 5

    avg_over=0
    s1,s2,s3,s4,s5=0
    for i in range(len(pred_list)):
        load_add = sub_add + pred_list[i]
        pred,dates = pkl.load(open(load_add,'rb'))

        for k in range(len(pred)):

            for s in range(5):

                if np.max(pred[k,s,0,:])>np.max(pred[k,s,-1,:]):
                    overall_larger+=1
                under_larger+=1

                a1 = np.max(pred[k,s,0,:])
                a2 = np.max(pred[k,s,1,:])
                a3 = np.max(pred[k,s,2,:])
                a4 = np.max(pred[k,s,3,:])
                a5 = np.max(pred[k,s,4,:])
                first_avg = (a1+a2+a3)/3.
                second_avg = (a3+a4+a5)/3.

                # Also want to calculate the standard deviation of these predictions
                s1+=np.std(pred[k,s,0])/(len(pred_list)*len(pred))
                s2+=np.std(pred[k,s,1])/(len(pred_list)*len(pred))
                s3+=np.std(pred[k,s,2])/(len(pred_list)*len(pred))
                s4+=np.std(pred[k,s,3])/(len(pred_list)*len(pred))
                s5+=np.std(pred[k,s,4])/(len(pred_list)*len(pred))
                if first_avg>second_avg:
                    avg_over+=1




    per_five = 100*(overall_larger/float(under_larger))
    avg_five = 100*(avg_over/float(under_larger))
    print('%.4f Percent First Max Bigger Than Final Max' %per_five)
    print('%.4f Percent First Average Larger Than The Second' %avg_five)
    print('Here are the standard deviations in prediction order:')
    print('%.4f' %s1)
    print('%.4f' %s2)
    print('%.4f' %s3)
    print('%.4f' %s4)
    print('%.4f' %s5)

    main_add = '/home/mharmon/FinanceProject/ModelResults/tickseq30'

    # Get out all of the 5 predictions first and calculate the number of neutral predictions vs. non-neutral
    # Also how often is it the exact same predictions?

    sub_add = main_add + 'win10/Predictions/'
    pred_list = os.listdir(sub_add)

    # The goal here is to look at the average max values
    overall_larger = 0
    under_larger =0
    window = 10

    avg_over=0
    for i in range(len(pred_list)):
        load_add = sub_add + pred_list[i]
        pred,dates = pkl.load(open(load_add,'rb'))

        for k in range(len(pred)):

            for s in range(5):

                if np.max(pred[k,s,0,:])>np.max(pred[k,s,-1,:]):
                    overall_larger+=1
                under_larger+=1

                a1 = np.max(pred[k,s,0,:])
                a2 = np.max(pred[k,s,1,:])
                a3 = np.max(pred[k,s,2,:])
                a4 = np.max(pred[k,s,3,:])
                a5 = np.max(pred[k,s,4,:])
                a6 = np.max(pred[k,s,5,:])
                a7 = np.max(pred[k,s,6,:])
                a8 = np.max(pred[k,s,7,:])
                a9 = np.max(pred[k,s,8,:])
                a10 = np.max(pred[k,s,9,:])
                first_avg = (a1+a2+a3+a4+a5)/5.
                second_avg = (a6+a7+a8+a9+a10)/5.

                if first_avg>second_avg:
                    avg_over+=1


    per_ten = 100*(overall_larger/float(under_larger))
    avg_ten = 100*(avg_over/float(under_larger))
    print('Now For Ten We Have The Following...')
    print('%.4f Percent First Max Bigger Than Final Max' %per_ten)
    print('%.4f Percent First Average Larger Than The Second' %avg_ten)
    return

# This function is for calculating the standard deviatoin across predictions
# I also need to calculate the entropy to see if there is a pattern here as well...

def main():

    main_add = '/home/mharmon/FinanceProject/ModelResults/tickconv10win5/Predictions/predictions'

    data_add = '/home/mharmon/FinanceProject/Data/tickdata/train10cnn0.pkl'
    winnum =5
    data,labels,dates = pkl.load(open(data_add,'rb'))
    del data
    del dates

    # Kull_back Leibler
    week_len = int(8064/4.)
    end = 8064

    labels = np.swapaxes(labels,1,2)

    std_dev = np.zeros((5,10)) # For five stock and 5 time periods
    divider=0
    top = 0
    kl_top = 0
    tv_top=0
    ent_top = 0
    for i in range(30):
        pred_add = main_add + str(i)+'.pkl'
        pred,chunk = pkl.load(open(pred_add,'rb'))
        del chunk
        test_labels = labels[end:end+week_len]
        # Sequence
        end = end+week_len
        for j in range(len(pred)):
            #Stock
            for k in range(len(pred[0,:,0,0])):

                # Time
                for m in range(len(pred[0,0,:,0])-1):

                    # Standard Deviation
                    std1 =np.std(pred[j,k,m,:])
                    std2 = np.std(pred[j,k,m+1,:])

                    # Kullback
                    kl_1 = kullback_leibler_divergence(test_labels[j,k,m,:],pred[j,k,m,:])
                    kl_2 = kullback_leibler_divergence(test_labels[j,k,m+1,:],pred[j,k,m+1,:])

                    # TV Values
                    tv_1 = total_variation_distance(test_labels[j,k,m,:],pred[j,k,m,:])
                    tv_2 = total_variation_distance(test_labels[j,k,m+1,:],pred[j,k,m+1,:])

                    ent_1 = crossentropy(test_labels[j,k,m,:],pred[j,k,m,:])
                    ent_2 = crossentropy(test_labels[j,k,m+1,:],pred[j,k,m+1,:])

                    divider+=1
                    if std1>std2 + np.finfo(float).eps:
                        top +=1
                    if kl_1>kl_2 +np.finfo(float).eps:
                        kl_top+=1
                    if tv_1>tv_2 + np.finfo(float).eps:
                        tv_top+=1
                    if ent_1>ent_2 + np.finfo(float).eps:
                        ent_top+=1



    print('Standard Deviation Percentage one compared to next:.')
    print(top/float(divider))
    print()
    print('KL Value Percentane onc compared to next:')
    print(kl_top/float(divider))
    print()
    print('Total Variation one compared to next:')
    print(tv_top/float(divider))
    print()
    print('Cross entropy compared to next:')
    print(ent_top/float(divider))


    # Now I need to calculate the KL divergence

    kl_store = np.zeros((5,winnum))
    for i in range(30):

        test_labels = labels[end:end+week_len]
        pred_add = main_add + str(i)+'.pkl'
        pred,chunk = pkl.load(open(pred_add,'rb'))
        del chunk
        end=end+week_len
        # For each time frame of predictions
        for j in range(5):

            for k in range(winnum):

                kl_store[j,k] += np.average(kullback_leibler_divergence(test_labels[:,j,k],pred[:,j,k]))

    kl_store/=30.


    print('KL average values:')
    print(kl_store)

    # Let's check percentages...
    week_len = int(8064/4.)
    end = 8064

    labels = np.swapaxes(labels,1,2)
    perc_top=0
    perc_bottom=0
    top_total=0
    cheb_total=0
    entropy_total=0
    temp_cheb = np.zeros(winnum)
    for i in range(30):

        test_labels = labels[end:end+week_len]
        pred_add = main_add + str(i)+'.pkl'
        pred,chunk = pkl.load(open(pred_add,'rb'))
        del chunk

        # For each time frame of predictions

        # Example
        end = end+week_len
        for m in range(len(pred)):

            # Stock
            for j in range(5):

                temp_storage = kullback_leibler_divergence(test_labels[m,j,:],pred[m,j,:])

                first_avg = np.average(temp_storage[0:5])
                second_avg = np.average(temp_storage[5:10])

                temp_var = total_variation_distance(test_labels[m,j,:],pred[m,j,:])
                first_var = np.average(temp_var[0:5])
                second_var = np.average(temp_var[5:10])

                temp_ent = crossentropy(test_labels[m,j,:],pred[m,j,:])
                first_ent = np.average(temp_var[0:5])
                second_ent = np.average(temp_var[0:5])
                for c in range(winnum):
                    temp_cheb[c] = scipy.spatial.distance.chebyshev(test_labels[m,j,c],pred[m,j,c])

                first_cheb = np.average(temp_cheb[0:5])
                second_cheb = np.average(temp_cheb[5:10])
                if first_avg>second_avg+np.finfo(float).eps:
                    perc_top+=1
                if first_var>second_var+np.finfo(float).eps:
                    top_total+=1
                if first_cheb>second_cheb+np.finfo(float).eps:
                    cheb_total+=1
                if first_ent > second_ent + np.finfo(float).eps:
                    entropy_total+=1
                perc_bottom+=1


    print('KL divergence of first average vs second:')
    print(perc_top/float(perc_bottom))
    print()
    print('TV of first average vs second')
    print(top_total/float(perc_bottom))
    print()
    print('Cheb value of first average vs second')
    print(cheb_total/perc_bottom)
    print()
    print('Entropy Total is the following')
    print(entropy_total/perc_bottom)


    # Some more calculations that Diego wanted done
    # Compare prediction of t to prediction of t+1

    sub_add = main_add + 'win5/Predictions/'
    pred_list = os.listdir(sub_add)

    # Predictions shape is (examples,stocks,time,class)
    divider=0
    kl_topper=0
    tv_top=0
    for i in range(len(pred_list)):
        load_add = sub_add + pred_list[i]
        pred,dates = pkl.load(open(load_add,'rb'))

        # Sample
        for k in range(len(pred)):
            #Stock
            for s in range(5):
                #window
                for w in range(3):
                    divider+=1

                    # KL Divergence
                    kl_preds1 = kullback_leibler_divergence(pred[k,s,w+1,:],pred[k,s,w,:])
                    kl_preds2 = kullback_leibler_divergence(pred[k,s,w+2,:],pred[k,s,w+1,:])

                    # Total Variation
                    tv_1 = total_variation_distance(pred[k,s,w+1,:],pred[k,s,w,:])
                    tv_2 = total_variation_distance(pred[k,s,w+2,:],pred[k,s,w+1,:])

                    if kl_preds1>kl_preds2 + np.finfo(float).eps:
                        kl_topper+=1
                    if tv_1 > tv_2 + np.finfo(float).eps:
                        tv_top+=1



    return
if __name__=='__main__':

    main()