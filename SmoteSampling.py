# Author: Mark Harmon
# Purpose: Smote function for my neural networks

# I can put all of my various SMOTE functions here, call on them when necessary
import numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE as smote

def regular_smote_seq(data,lab,seqsize,window,stocks,classes,majority):

    neutral = majority
    totallabel = [[],[],[],[],[]]
    for i in range(stocks):

        templab = np.argmax(lab[i][:,0,:],axis=1)
        largestvec = np.where(templab==neutral)[0]
        tempdata = []
        templabel = [[], [], [], [], []]
        for j in range(classes):

            if j!=neutral:
                othervec = np.where(templab==j)[0]


                if len(othervec)>0 and (len(othervec)/len(largestvec)<0.5):
                    # Here I need to find a label that occurs at least twice.  Any of them will do...
                    factor = 1
                    vals = np.zeros(len(othervec))

                    for m in range(stocks):
                        # Need to make a unique identifier for each that's quick to do.  Since the list are pretty small,
                        # I can just multiple by multiples of ten.  This will uniquely identify them.
                        if m!=i:
                            vals += factor*(np.argmax(lab[m][:,0,:],axis=1)[othervec]+1)
                            factor*=10

                    myuns = np.unique(vals)

                    # Find where there is a label that appears more than twice...
                    uniquelabels = []
                    smoteratio=[]
                    for m in range(len(myuns)):
                        onevec = np.where(myuns[m]==vals)[0]
                        if len(onevec)>=5:
                            uniquelabels +=[onevec]
                            smoteratio.append(len(onevec))
                    smoteratio = np.array(smoteratio)

                    smoteratio = 0.5*(smoteratio/np.sum(smoteratio))
                    if len(uniquelabels)>0:
                        for k in range(len(uniquelabels)):


                            myvec = uniquelabels[k]
                            uselabel = [lab[m][othervec[myvec[0]]] for m in range(5)]
                            tempvec = np.append(largestvec,np.array(othervec[myvec]))
                            fitlab = templab[tempvec]
                            fitlab[fitlab != neutral] = 0
                            fitlab[fitlab==neutral]=1
                            fitdat = np.reshape(data[tempvec],(len(data[tempvec]),seqsize*5))
                            sme = smote(ratio = smoteratio[k],random_state=42,k_neighbors=len(othervec[myvec])-1)
                            uselabel = np.array([lab[m][tempvec] for m in range(5)])
                            uselabel = np.swapaxes(uselabel,0,1)
                            uselabel = np.reshape(uselabel,(len(uselabel),len(uselabel[0,:,0,0])*len(uselabel[0,0,:,0])*len(uselabel[0,0,0,:])))

                            datanew,chunk = sme.fit_sample(fitdat,fitlab)
                            labnew,chunk = sme.fit_sample(uselabel,fitlab)

                            take = np.where(chunk ==0)[0]
                            datanew = datanew[take]
                            labnew = labnew[take]
                            takelen = len(data[myvec])

                            datanew = datanew[takelen:]
                            labnew = labnew[takelen:]
                            labnew = np.reshape(labnew,(len(labnew),stocks,window,classes))
                            truearg = np.argmax(labnew,axis=3)
                            truelab = np.zeros((labnew.shape))

                            for ex in range(len(truelab)):

                                for s in range(len(truelab[0,:,0,0])):

                                    for time in range(len(truelab[0,0,:,0])):
                                            truelab[ex,s,time,truearg[ex,s,time]]=1

                            truelab = np.swapaxes(truelab,0,1)


                            if len(tempdata)==0:
                                tempdata = datanew
                                for m in range(stocks):
                                        templabel[m] = np.array(truelab[m])

                            else:
                                tempdata = np.vstack((tempdata,datanew))

                                for m in range(stocks):
                                    templabel[m]=np.vstack((templabel[m],truelab[m]))


        if len(tempdata)>0:# Since data is all in one single stack
            if i==0:
                totaldata = tempdata
                for m in range(stocks):
                    totallabel[m] = templabel[m]
            else:
                totaldata = np.vstack((totaldata,tempdata))
                for m in range(stocks):
                    totallabel[m] = np.vstack((totallabel[m],templabel[m]))


    totaldata = np.reshape(totaldata,(len(totaldata),seqsize,5))
    totaldata = np.vstack((totaldata,data))

    for m in range(stocks):
        totallabel[m] = np.vstack((totallabel[m],lab[m]))


    # Don't forget that I'm going to have to mix the data up quite a bit as well.
    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

def smote_concat_seq(data,lab,seqsize,window,stocks,num_class):

    # label must be of the following form: (stocks,examples,sequence,classes)
    totallabel = [[] for i in range(stocks)]


    all_labels=np.zeros((len(lab[0])))
    factor=1
    for j in range(stocks):
        all_labels+=factor*(np.argmax(lab[j][:,0,:],axis=1)+1)
        factor*=10


    # Get all unique labels that are big enough
    # Also determine the majority
    majority = stats.mode(all_labels)[0][0]
    unique_labels = np.unique(all_labels)
    majority_labels = np.where(all_labels==majority)[0]
    minority_labels = np.delete(unique_labels,np.where(unique_labels==majority)[0])
    fit_labels=[]
    smoteratio = []
    for m in range(len(minority_labels)):
        onevec = np.where(minority_labels[m]==all_labels)[0]
        if len(onevec) >= 5:
            fit_labels += [onevec]

    tempdata = []
    templabel = [[] for i in range(stocks)]
    if len(fit_labels) > 0:
        for k in range(len(fit_labels)):

            if len(fit_labels[k])/(float(len(majority_labels)))<0.5:
                all_vec = np.append(majority_labels,fit_labels[k])
                binary_labels = np.zeros((all_vec.shape))
                binary_labels[len(majority_labels):]=1
                smote_dat = data[all_vec]
                smote_dat = np.reshape(smote_dat,(len(smote_dat),seqsize*stocks))
                smote_lab = np.swapaxes(np.array([lab[m][all_vec] for m in range(stocks)]),0,1)
                smote_lab = np.reshape(smote_lab,(len(smote_lab),window*stocks*num_class))

                smote_all = np.concatenate((smote_dat,smote_lab),axis=1)
                sme = smote(ratio=1., random_state=42, k_neighbors=len(fit_labels[k])-1)


                temp_new_dat,chunk = sme.fit_sample(smote_all,binary_labels)

                # Seperate data from labels
                new_dat = temp_new_dat[:,0:smote_dat.shape[1]]
                new_lab = temp_new_dat[:,smote_dat.shape[1]:]

                # Seperate new from old
                new_dat = new_dat[len(smote_all):]
                new_dat = np.reshape(new_dat,(len(new_dat),seqsize,stocks))
                new_lab = new_lab[len(smote_all):]
                new_lab = np.reshape(new_lab,(len(new_lab),stocks,window,num_class))

                actual_lab = np.zeros((new_lab.shape))
                new_lab = np.argmax(new_lab,axis=3)

                for a in range(len(actual_lab)):

                    for b in range(stocks):

                        for c in range(window):

                            actual_lab[a,b,c,new_lab[a,b,c]]=1


                if len(tempdata) == 0:
                    tempdata = new_dat
                    for m in range(stocks):
                        templabel[m] = np.array(actual_lab[:,m,:,:])

                else:
                    tempdata = np.vstack((tempdata, new_dat))

                    for m in range(stocks):
                        templabel[m] = np.vstack((templabel[m], actual_lab[:,m,:,:]))



    totaldata = np.vstack((data,tempdata))

    for i in range(stocks):
        totallabel[i] = np.vstack((lab[i],templabel[i]))


    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

def smote_binary_seq(data,lab,seqsize,stocks):

    # label must be of the following form: (stocks,examples,sequence,classes)
    totallabel = [[] for i in range(stocks)]


    all_labels=np.zeros((len(lab[0])))
    factor=1
    for j in range(stocks):
        all_labels+=factor*(lab[j][:,0]+1)
        factor*=10


    # Get all unique labels that are big enough
    # Also determine the majority
    majority = stats.mode(all_labels)[0][0]
    unique_labels = np.unique(all_labels)
    majority_labels = np.where(all_labels==majority)[0]
    minority_labels = np.delete(unique_labels,np.where(unique_labels==majority)[0])
    fit_labels=[]
    for m in range(len(minority_labels)):
        onevec = np.where(minority_labels[m]==all_labels)[0]
        if len(onevec) >= 5:
            fit_labels += [onevec]

    tempdata = []
    templabel = [[] for i in range(stocks)]
    if len(fit_labels) > 0:
        for k in range(len(fit_labels)):

            if len(fit_labels[k])/(float(len(majority_labels)))<0.5:
                all_vec = np.append(majority_labels,fit_labels[k])
                binary_labels = np.zeros((all_vec.shape))
                binary_labels[len(majority_labels):]=1
                smote_dat = data[all_vec]
                smote_dat = np.reshape(smote_dat,(len(smote_dat),seqsize*stocks))
                smote_lab = np.swapaxes(np.array([lab[m][all_vec] for m in range(stocks)]),0,1)
                smote_lab = np.reshape(smote_lab,(len(smote_lab),stocks))

                smote_all = np.concatenate((smote_dat,smote_lab),axis=1)
                sme = smote(ratio=1., random_state=42, k_neighbors=len(fit_labels[k])-1,kind='svm')


                temp_new_dat,chunk = sme.fit_sample(smote_all,binary_labels)

                # Seperate data from labels
                new_dat = temp_new_dat[:,0:smote_dat.shape[1]]
                new_lab = temp_new_dat[:,smote_dat.shape[1]:]

                # Seperate new from old
                new_dat = new_dat[len(smote_all):]
                new_dat = np.reshape(new_dat,(len(new_dat),seqsize,stocks))
                new_lab = new_lab[len(smote_all):]
                new_lab = np.reshape(new_lab,(len(new_lab),stocks,1))

                actual_lab = np.round(new_lab)


                if len(tempdata) == 0:
                    tempdata = new_dat
                    for m in range(stocks):
                        templabel[m] = np.array(actual_lab[:,m,:])

                else:
                    tempdata = np.vstack((tempdata, new_dat))

                    for m in range(stocks):
                        templabel[m] = np.vstack((templabel[m], actual_lab[:,m,:]))



    totaldata = np.vstack((data,tempdata))

    for i in range(stocks):
        totallabel[i] = np.vstack((lab[i],templabel[i]))


    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

def smote_conv_seq(data,lab,seqsize,window,stocks,num_class):

    # label must be of the following form: (stocks,examples,sequence,classes)
    totallabel = [[] for i in range(stocks)]


    all_labels=np.zeros((len(lab[0])))
    factor=1
    for j in range(stocks):
        all_labels+=factor*(np.argmax(lab[j][:,0,:],axis=1)+1)
        factor*=10


    # Get all unique labels that are big enough
    # Also determine the majority
    majority = stats.mode(all_labels)[0][0]
    unique_labels = np.unique(all_labels)
    majority_labels = np.where(all_labels==majority)[0]
    minority_labels = np.delete(unique_labels,np.where(unique_labels==majority)[0])
    fit_labels=[]
    smoteratio = []
    for m in range(len(minority_labels)):
        onevec = np.where(minority_labels[m]==all_labels)[0]
        if len(onevec) >= 5:
            fit_labels += [onevec]

    tempdata = []
    templabel = [[] for i in range(stocks)]
    if len(fit_labels) > 0:
        for k in range(len(fit_labels)):

            if (len(fit_labels[k])/(float(len(majority_labels)))<0.5):
                all_vec = np.append(majority_labels,fit_labels[k])
                binary_labels = np.zeros((all_vec.shape))
                binary_labels[len(majority_labels):]=1
                smote_dat = data[all_vec]
                smote_dat = np.reshape(smote_dat,(len(smote_dat),4*seqsize*stocks))
                smote_lab = np.swapaxes(np.array([lab[m][all_vec] for m in range(stocks)]),0,1)
                smote_lab = np.reshape(smote_lab,(len(smote_lab),window*stocks*num_class))

                smote_all = np.concatenate((smote_dat,smote_lab),axis=1)
                sme = smote(ratio=1., random_state=42, k_neighbors=len(fit_labels[k])-1)


                temp_new_dat,chunk = sme.fit_sample(smote_all,binary_labels)

                # Seperate data from labels
                new_dat = temp_new_dat[:,0:smote_dat.shape[1]]
                new_lab = temp_new_dat[:,smote_dat.shape[1]:]

                # Seperate new from old
                new_dat = new_dat[len(smote_all):]
                new_dat = np.reshape(new_dat,(len(new_dat),4,stocks,seqsize,1))
                new_lab = new_lab[len(smote_all):]
                new_lab = np.reshape(new_lab,(len(new_lab),stocks,window,num_class))

                actual_lab = np.zeros((new_lab.shape))
                new_lab = np.argmax(new_lab,axis=3)

                for a in range(len(actual_lab)):

                    for b in range(stocks):

                        for c in range(window):

                            actual_lab[a,b,c,new_lab[a,b,c]]=1


                if len(tempdata) == 0:
                    tempdata = new_dat
                    for m in range(stocks):
                        templabel[m] = np.array(actual_lab[:,m,:,:])

                else:
                    tempdata = np.vstack((tempdata, new_dat))

                    for m in range(stocks):
                        templabel[m] = np.vstack((templabel[m], actual_lab[:,m,:,:]))



    totaldata = np.vstack((data,tempdata))

    for i in range(stocks):
        totallabel[i] = np.vstack((lab[i],templabel[i]))


    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

def smote_conv_many_stock(data,lab,seqsize,window,stocks,num_class):

    # label must be of the following form: (stocks,examples,sequence,classes)
    totallabel = [[] for i in range(stocks)]


    all_labels=np.zeros((len(lab[0])))
    factor=0.000001
    for j in range(stocks):
        all_labels+=factor*(np.argmax(lab[j][:,0,:],axis=1)+1)
        factor*=10


    # Now find all unique labels...
    # Get all unique labels that are big enough
    # Also determine the majority
    majority = stats.mode(all_labels)[0][0]
    unique_labels = np.unique(all_labels)
    majority_labels = np.where(all_labels==majority)[0]
    minority_labels = np.delete(unique_labels,np.where(unique_labels==majority)[0])
    fit_labels=[]
    smoteratio = []
    for m in range(len(minority_labels)):
        onevec = np.where(minority_labels[m]==all_labels)[0]
        if len(onevec) >= 2:
            fit_labels += [onevec]

    tempdata = []
    templabel = [[] for i in range(stocks)]
    if len(fit_labels) > 0:
        for k in range(len(fit_labels)):

            if (len(fit_labels[k])/(float(len(majority_labels)))<0.5):
                all_vec = np.append(majority_labels,fit_labels[k])
                binary_labels = np.zeros((all_vec.shape))
                binary_labels[len(majority_labels):]=1
                smote_dat = data[all_vec]
                smote_dat = np.reshape(smote_dat,(len(smote_dat),4*seqsize*stocks))
                smote_lab = np.swapaxes(np.array([lab[m][all_vec] for m in range(stocks)]),0,1)
                smote_lab = np.reshape(smote_lab,(len(smote_lab),window*stocks*num_class))

                smote_all = np.concatenate((smote_dat,smote_lab),axis=1)
                sme = smote(ratio=1., random_state=42, k_neighbors=len(fit_labels[k])-1)


                temp_new_dat,chunk = sme.fit_sample(smote_all,binary_labels)

                # Seperate data from labels
                new_dat = temp_new_dat[:,0:smote_dat.shape[1]]
                new_lab = temp_new_dat[:,smote_dat.shape[1]:]

                # Seperate new from old
                new_dat = new_dat[len(smote_all):]
                new_dat = np.reshape(new_dat,(len(new_dat),4,stocks,seqsize,1))
                new_lab = new_lab[len(smote_all):]
                new_lab = np.reshape(new_lab,(len(new_lab),stocks,window,num_class))

                actual_lab = np.zeros((new_lab.shape))
                new_lab = np.argmax(new_lab,axis=3)

                for a in range(len(actual_lab)):

                    for b in range(stocks):

                        for c in range(window):

                            actual_lab[a,b,c,new_lab[a,b,c]]=1


                if len(tempdata) == 0:
                    tempdata = new_dat
                    for m in range(stocks):
                        templabel[m] = np.array(actual_lab[:,m,:,:])

                else:
                    tempdata = np.vstack((tempdata, new_dat))

                    for m in range(stocks):
                        templabel[m] = np.vstack((templabel[m], actual_lab[:,m,:,:]))



    totaldata = np.vstack((data,tempdata))

    for i in range(stocks):
        totallabel[i] = np.vstack((lab[i],templabel[i]))


    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

def smote_conv_new(data,lab,seqsize,window,stocks,num_class):

    # label must be of the following form: (stocks,examples,sequence,classes)
    totallabel = [[] for i in range(stocks)]

    original_labels = np.zeros((len(lab[0]),stocks,window,num_class))
    for j in range(stocks):
        original_labels[:,j,:,:]=lab[j]

    all_labels = np.sum(original_labels,axis=0)
    sub_all = np.copy(all_labels)
    for j in range(stocks):
        all_labels[j]= all_labels[j]/np.max(all_labels[j])


    # Now find the one that most needs to be made larger that has the minimum number of points
    sorted_classes = np.sort(all_labels[:,0,:],axis=None)
    top_k = 10
    min_lab = 5.
    temp_class = np.where(sorted_classes >min_lab/np.max(sub_all[:,0,:]))[0]

    class_location=np.zeros((top_k,2))
    majority_location = np.zeros((top_k,2))
    for j in range(top_k):
        temp_location=np.where(sorted_classes[temp_class[j]]==all_labels[:,0,:])

        # for now, we are just goin to take the first one.
        class_location[j,0] = temp_location[0][0]
        class_location[j,1]=temp_location[1][0]

        majority_location[j,0] = temp_location[0][0]

        temp_majority = np.where(all_labels[temp_location[0][0],0,:]==np.max(all_labels[temp_location[0][0],0,:]))[0]
        majority_location[j,1]=temp_majority




    tempdata = []
    templabel = [[] for i in range(stocks)]
    for j in range(top_k):
        # Find fit_labels and majority_labels here...
        fit_labels = np.where(original_labels[:,int(class_location[j,0]),0,int(class_location[j,1])]==1)[0]
        majority_labels=np.where(original_labels[:,int(majority_location[j,0]),0,int(majority_location[j,1])]==1)[0]

        if (len(fit_labels)/(float(len(majority_labels)))<0.75):
            all_vec = np.append(majority_labels,fit_labels)
            binary_labels = np.zeros((all_vec.shape))
            binary_labels[len(majority_labels):]=1
            smote_dat = data[all_vec]
            smote_dat = np.reshape(smote_dat,(len(smote_dat),4*seqsize*stocks))
            smote_lab = np.swapaxes(np.array([lab[m][all_vec] for m in range(stocks)]),0,1)
            smote_lab = np.reshape(smote_lab,(len(smote_lab),window*stocks*num_class))

            smote_all = np.concatenate((smote_dat,smote_lab),axis=1)
            sme = smote(ratio='minority', random_state=42, k_neighbors=len(fit_labels)-1)


            temp_new_dat,chunk = sme.fit_sample(smote_all,binary_labels)

            # Seperate data from labels
            new_dat = temp_new_dat[:,0:smote_dat.shape[1]]
            new_lab = temp_new_dat[:,smote_dat.shape[1]:]

            # Seperate new from old
            new_dat = new_dat[len(smote_all):]
            new_dat = np.reshape(new_dat,(len(new_dat),4,stocks,seqsize,1))
            new_lab = new_lab[len(smote_all):]
            new_lab = np.reshape(new_lab,(len(new_lab),stocks,window,num_class))

            actual_lab = np.zeros((new_lab.shape))
            new_lab = np.argmax(new_lab,axis=3)

            for a in range(len(actual_lab)):

                for b in range(stocks):

                    for c in range(window):

                        actual_lab[a,b,c,new_lab[a,b,c]]=1


            if len(tempdata) == 0:
                tempdata = new_dat
                for m in range(stocks):
                    templabel[m] = np.array(actual_lab[:,m,:,:])

            else:
                tempdata = np.vstack((tempdata, new_dat))

                for m in range(stocks):
                    templabel[m] = np.vstack((templabel[m], actual_lab[:,m,:,:]))



    totaldata = np.vstack((data,tempdata))

    for i in range(stocks):
        totallabel[i] = np.vstack((lab[i],templabel[i]))


    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(stocks):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

