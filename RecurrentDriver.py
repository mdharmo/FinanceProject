# Author: Mark Harmon
# Purpose: Driver for recurrent network on the cme data

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from cmestateful import main


address = '/home/mharmon/FinanceProject/Data/cmenew/'

epoch = 50
batchsize = 100

results = main(address,epoch,batchsize)

# Desirable Figures:
# Confusion Matrix, we have really unbalanced classes.
