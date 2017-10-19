# Author: Mark Harmon
# Purpose: To make a training loss figure for different sequence sizes...

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

loss_1 = np.array([4.63285,4.53882,4.39274,4.31688,4.22495,4.14533,4.11012,4.076,4.042,4.032,4.,3.9656])

seq = np.array([5,10,15,20,25,30,35,40,45,50,55,60])
saveadd = '/home/mharmon/FinanceProject/ModelResults/SeqvsLossSameEpoch.png'
fig,ax = plt.subplots()
ax.scatter(seq,loss_1)
plt.xlabel('Sequence Length')
plt.ylabel('Training Loss')
plt.title('Sequence Length vs Training Loss (Same Epoch Count)')
plt.savefig(saveadd)
plt.close()

loss_2 = np.array([4.63285,4.53882,4.39274,4.31688,4.22495,4.14533,2.7245,3.9,3.86297,4.032,2.3519,2.6506])
epochs = np.array([6,6,6,6,6,6,15,7,7,6,17,15])
saveadd = '/home/mharmon/FinanceProject/ModelResults/SeqvsLossDiffEpoch.png'
fig,ax=plt.subplots()
ax.scatter(seq,loss_2)
for i in range(len(loss_2)):
    ax.annotate(epochs[i],(seq[i],loss_2[i]))
plt.xlabel('Sequence Length')
plt.ylabel('Training Loss')
plt.title('Sequence Length vs Training Loss (Diff Epoch Count)')
plt.savefig(saveadd)
plt.close()



