# Author: Mark Harmon
# Purpose: To test SMOTE

from imblearn.over_sampling import SMOTE as smote
import numpy as np
import pickle as pkl

def main():

    # Make fake data and labels first...
    data = np.random.rand(10000,10)
    labels = np.zeros((10000,5,5,5)) #(Ex,stock,time,class)

    # Make random labels
    for i in range(10000):
        for j in range(5):
            for k in range(5):
                labels[i,j,k,np.random.randint(0,5)]=1

    # Now implement smote on the data, and hopefully labels


    temp_label = np.argmax(labels[:,:,0,:],axis=2)

    place_1 = np.where(temp_label[:,0]==0)[0]
    place_2 = np.where(temp_label[:,0]==3)[0]


    if len(place_1)>len(place_2):
        smote_ratio = len(place_2)/float(len(place_1))
        minority_class  = place_2
        majority_class = place_1
        myvec = np.concatenate((place_1, place_2))
    else:
        smote_ratio = len(place_1)/float(len(place_2))
        minority_class = place_1
        majority_class = place_2
        myvec = np.concatenate((place_2, place_1))



    sme = smote(ratio=1., random_state=42, k_neighbors=len(minority_class) - 1)

    sme_lab = np.zeros((len(myvec)))
    sme_lab[0:len(majority_class)]=0
    sme_lab[len(majority_class):]=1

    datanew, chunk = sme.fit_sample(data[myvec], sme_lab)

    labels = np.reshape(labels,(10000,5*5*5))


    return

if __name__=='__main__':

    main()