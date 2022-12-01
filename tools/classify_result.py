import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def classify_result(result, label, thres=0.5, print_result=False):
    result_pd = pd.DataFrame(result)
    right = result_pd[((result_pd[0]<thres)&(label==0))|((result_pd[0]>thres)&(label==1))]
    TP = result_pd[(result_pd[0]>thres)&(label==1)]
    TN = result_pd[(result_pd[0]<thres)&(label==0)]
    FP = result_pd[(result_pd[0]>thres)&(label==0)]
    FN = result_pd[(result_pd[0]<thres)&(label==1)]
    if print_result:
        figure = plt.figure(0)
        plt.scatter(np.array(right.index),right.values[:,0],color='g',marker='.',label='Right Predicted')
        plt.scatter(np.array(FP.index),FP.values[:,0],color='orange', marker='.',label='False Positive') # 'orange'
        plt.scatter(np.array(FN.index),FN.values[:,0],color='salmon', marker='.',label='False Negative') # 'salmon'
        plt.xlabel('Sample Number', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.ylim([-0.05,1.3])
        plt.yticks(np.arange(0,1.3,1))
        plt.show()
    return right, TP, TN, FP, FN