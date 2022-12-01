# Generate the convolution dictionary according the network paramater
import numpy as np
class GenConvDict():
    def __init__(self, args, ParaDict, MatrixName):
        super(GenConvDict, self).__init__()
        self.args = args
        self.SigLengthList = [1024]
        self.StrideList = args.SL
        self.PadList = args.PL
        self.ParaDict = ParaDict
        self.MatrixName = MatrixName
    def GenConvMatrix(self, transpose=False):
        ConvMatrix = {}
        for m, Name in enumerate(self.MatrixName):
        # 参数初始化
            SigLength = self.SigLengthList[-1]
            out_channel,in_channel,kernal_length = self.ParaDict[Name].shape[0], self.ParaDict[Name].shape[1], self.ParaDict[Name].shape[2]
            stride,padding = self.StrideList[m],self.PadList[m]
            slide = int((SigLength-kernal_length+2*padding)/stride)+1
            self.SigLengthList.append(slide)
            # 卷积矩阵计算
            con_matrix = np.zeros((in_channel*SigLength,slide*out_channel))
            for i in range(in_channel):
                for j in range(out_channel):
                    for k in range(slide):
                        beg_index = i*SigLength+k*stride-padding
                        end_index = i*SigLength+k*stride-padding+kernal_length
                        if beg_index<i*SigLength:
                            con_matrix[i*SigLength:end_index,j*slide+k] = self.ParaDict[Name][j,i,i*SigLength-beg_index:]
                        elif end_index>(i+1)*SigLength:
                            con_matrix[beg_index:(i+1)*SigLength,j*slide+k] = self.ParaDict[Name][j,i,:kernal_length-end_index+(i+1)*SigLength]
                        else:
                            con_matrix[beg_index:end_index,j*slide+k] = self.ParaDict[Name][j,i]
            if transpose:
                ConvMatrix.update({Name:con_matrix.T})
            else:
                ConvMatrix.update({Name:con_matrix})
        return ConvMatrix, self.SigLengthList
    
    def CalKernelLen(self):
        KernelList = self.args.KL
        StrideList = self.args.SL
        KernelLength = [KernelList[0]]
        StrideLength = [StrideList[0]]
        for KerLen, Stride in zip(KernelList[1:], StrideList[1:]):
            KernelLength.append(KernelLength[-1]+(KerLen-1)*StrideLength[-1])
            StrideLength.append(Stride*StrideLength[-1])
        return KernelLength, StrideLength