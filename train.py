"""
The code for paper 'Adversarial Algorithm Unrolling Network for Interpretable Mechanical Anomaly Detection'

If you want do research based this code, please cite the above paper.

For any questions you can contact e-mail: Albert_An@foxmail.com

Copyright reserved by the authors

"""

from model.AAU_Net import parse_args, train_AAUNet
from tools.SeedSetting import setup_seed

def train(seed):
    
    # setting ramdom seed
    setup_seed(seed)

    # setting paramater
    opt = parse_args()

    # train model
    ACC = train_AAUNet(opt)
    
    return ACC

if __name__ == '__main__':
    ACC = train(8)