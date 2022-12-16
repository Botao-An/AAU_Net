# AAU-Net

1. We provide the main code for paper "**Adversarial Algorithm Unrolling Network for Interpretable Mechanical Anomaly Detection**".

   

2. The code is originally debugged on the computer with the following configuration.

   |          Hardware           |    Software    |     Software      |
   | :-------------------------: | :------------: | :---------------: |
   |  Intel Core i7-10700KF CUP  | Anaconda 4.9.2 |     CUDA 11.7     |
   |          RAM 32GB           |   Python 3.8   |    cuDNN 8.4.0    |
   | NVIDIA GeForce RTX 3080 GPU | PyTorch 1.8.1  | TorchVision 0.9.1 |



3. We give a demo of simulation for you to test the performance of AAU-Net initially.

   

4. To run the model, you should firstly prepare the dataset in a standard format as the simulated one. More details can be found in file **pre and post processing.ipynb**. Then you can run AAU-Net by file **train.py**. To check and visulize the results, you can use **pre and post processing.ipynb** again. 

   
   
5. If you want do research based this code, please cite the following papers. For any questions you can contact e-mail: Albert_An@foxmail.com. 
   ```latex
   @article{an2022interpretable,
      title={Interpretable Neural Network via Algorithm Unrolling for Mechanical Fault Diagnosis},
      author={An, Botao and Wang, Shibin and Zhao, Zhibin and Qin, Fuhua and Yan, Ruqiang and Chen, Xuefeng},
      journal={IEEE Transactions on Instrumentation and Measurement},
      volume={71},
      pages={1--11},
      year={2022},
      publisher={IEEE}
   }
   @article{an2022adversarial,
      title={Adversarial Algorithm Unrolling Network for Interpretable Mechanical Anomaly Detection},
      author={An, Botao and Wang, Shibin and Qin, Fuhua and Zhao, Zhibin and Yan, Ruqiang and Chen, Xuefeng},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      volume={},
      pages={},
      year={2022},
      publisher={IEEE}
   }
   ```
