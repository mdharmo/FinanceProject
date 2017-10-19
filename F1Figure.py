# Author: Mark Harmon
# Purpose: Make F1 Figure for finance tick data


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


# Let's just plot the first 20 weeks.
folderlist = ['tick5','tick10','tick15','tick20']
color = ['bo','go','ro','ko','bs','gs','rs','ks']

add1 = '/home/mharmon/FinanceProject/ModelResults/'
saveadd = '/home/mharmon/FinanceProject/ModelResults/F1PlotDiffSeq0.png'

for i in range(len(folderlist)):

    f1add = add1 + folderlist[i] + '/f1avg.npy'
    f1data = np.load(f1add)
    week = np.arange(1,31)
    plt.figure(0)
    plt.plot(week,f1data,color[i],linestyle='solid')
    print(np.mean(f1data))

plt.xlabel('Week')
plt.ylabel('Average F1 Value')
plt.title('Average F1 Values with Various Models')
plt.legend(labels = ['5 Seq','10 Seq', '15 Seq', '20 Seq'])
plt.savefig(saveadd)
plt.close()

# Now for the second figure
folderlist = ['tick25','tick30','tick35','tick40']
saveadd = '/home/mharmon/FinanceProject/ModelResults/F1PlotDiffSeq1.png'
for i in range(len(folderlist)):

    f1add = add1 + folderlist[i] + '/f1avg.npy'
    f1data = np.load(f1add)
    week = np.arange(1, 31)
    plt.figure(1)
    plt.plot(week, f1data, color[i], linestyle='solid')
    print(np.mean(f1data))

plt.xlabel('Week')
plt.ylabel('Average F1 Value')
plt.title('Average F1 Values with Various Models')
plt.legend(labels=['25 Seq', '30 Seq', '35 Seq', '40 Seq'])
plt.savefig(saveadd)
plt.close()

