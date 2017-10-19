# Author: Mark Harmon
# Purpose: Test the sampled predictions to see if patters arise...

import numpy as np
import pickle as pkl
import scipy.stats
def main():
    main_add_1 = '/home/mharmon/FinanceProject/ModelResults/tickconvsamp10win5/Predictions/predictions'
    main_add_2 = '/home/mharmon/FinanceProject/ModelResults/tickconvdrop10win5/Predictions/predictions'

    denom1 = 0
    not_same1=0
    denom2 = 0
    not_same2 = 0
    for i in range(11):

        # Take first prediction here
        pred_add_z = main_add_1 + str(i)+'run0.pkl'
        pred_orig_z,chunk = pkl.load(open(pred_add_z,'rb'))
        pred_orig_z = np.argmax(pred_orig_z,axis=3)

        pred_add_drop = main_add_2 + str(i)+'run0.pkl'
        pred_orig_drop,chunk = pkl.load(open(pred_add_drop,'rb'))
        pred_orig_drop = np.argmax(pred_orig_drop,axis=3)

        for j in range(1,30):
            # For z-sampling method
            pred_add_z = main_add_1 + str(i)+'run'+str(j)+'.pkl'
            pred_new_z,chunk = pkl.load(open(pred_add_z,'rb'))
            pred_new_z = np.argmax(pred_new_z,axis=3)

            not_same1 += len(np.where(pred_orig_z!=pred_new_z)[0])
            denom1 +=2016*5*5.

            # For dropout sampling method

            pred_add_drop = main_add_2 + str(i)+'run'+str(j)+'.pkl'
            pred_new_drop,chunk = pkl.load(open(pred_add_drop,'rb'))
            pred_new_drop = np.argmax(pred_new_drop,axis=3)

            not_same2 += len(np.where(pred_orig_drop!=pred_new_drop)[0])
            denom2 +=2016*5*5.


    value1 = 100*(not_same1)/denom1
    value2 = 100*(not_same2)/denom2
    print('Percentage for Sampling Method')
    print(value1)
    print('Percentage for Dropout')
    print(value2)

    # Here is where I check to make sure that the confidence lowers through time..

    denom1 = 0
    denom2 =0
    top1=0
    top2=0
    avg_table_z = np.zeros((5,5))
    avg_table_drop = np.zeros((5,5))
    for i in range(11):

        for j in range(30):

            # For each time step, calculate the percentage.
            # It's probably better to do a count method than
            # For z-sampling method
            pred_add_z = main_add_1 + str(i)+'run'+str(j)+'.pkl'
            pred_new_z,chunk = pkl.load(open(pred_add_z,'rb'))
            pred_new_z = np.argmax(pred_new_z,axis=3)


            # For dropout sampling method

            pred_add_drop = main_add_2 + str(i)+'run'+str(j)+'.pkl'
            pred_new_drop,chunk = pkl.load(open(pred_add_drop,'rb'))
            pred_new_drop = np.argmax(pred_new_drop,axis=3)

            avg_table_z += scipy.stats.mode(pred_new_z)[1][0]/(30*2016*11)
            avg_table_drop += scipy.stats.mode(pred_new_drop)[1][0]/(30*2016*11)
            # For each stock
            for k in range(5):

                most_chosen_z = scipy.stats.mode(pred_new_z[:,k,:])[0][0]
                most_chosen_drop = scipy.stats.mode(pred_new_drop[:,k,:])[0][0]


                # For each time period
                for n in range(4):

                    most_z_percentage = len(np.where(pred_new_z[:,k,n]==most_chosen_z[n])[0])\
                                        /float(len(pred_new_z))
                    most_drop_percentage = len(np.where(pred_new_drop[:,k,n]==most_chosen_drop[n])[0])\
                                           /float(len(pred_new_drop))

                    most_z_percentage_next = len(np.where(pred_new_z[:,k,n+1]==most_chosen_z[n])[0])\
                                             /float(len(pred_new_z))
                    most_drop_percentage_next = len(np.where(pred_new_drop[:,k,n+1]==most_chosen_drop[n])[0])\
                                                /float(len(pred_new_drop))

                    if most_z_percentage>= most_z_percentage_next + np.finfo(float).eps:
                        top1+=1
                    if most_drop_percentage>= most_drop_percentage_next + np.finfo(float).eps:
                        top2+=1

                    denom1+=1
                    denom2+=1

    value1 = 100*top1/float(denom1)
    value2 = 100*top2/float(denom2)
    print()
    print('Percentage of time confidence decreases in time during sampling method')
    print(value1)
    print('Percentage of time confidence decreases in time during drop method')
    print(value2)
    print()
    print('Table for z')
    print(100*avg_table_z)
    print('Table for drop')
    print(100*avg_table_drop)

    # Do averages from one table to another...
    first_averages_z = np.average(avg_table_z[:,0:3],axis=1)
    second_averages_z = np.average(avg_table_z[:,2:5],axis=1)

    first_averages_drop = np.average(avg_table_drop[:,0:3],axis=1)
    second_averages_drop = np.average(avg_table_drop[:,2:5],axis=1)

    print('Compare table averages')
    print(first_averages_z,second_averages_z)
    print()
    print()
    print(first_averages_drop,second_averages_drop)
    return

if __name__=='__main__':
    main()