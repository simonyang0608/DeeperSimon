#*********************************************************************************#
# Source: Module.py                                                               #
#                                                                                 #
# Description: Customized block functions to build model backbone/encoder/decoder #
#                                                                                 #
# Author: SimonYang                                                               #
#*********************************************************************************#

#================#
# Import Section #
#================#
#############################
#Pytorch max, mean, cat, ...
from torch import (max as torch_max, mean as torch_mean, \
                   cat as torch_cat, argmax as torch_argmax)

############################################################
#Pytorch nn module (i.e. basic inherit), sequential, tensor
from Model.Global_Builder.Unit import (Module, Sequential, Tensor)

############################################################
#Pytorch nn pooling layer unit (i.e. maxpool, avgpool, ...)
from torch.nn import MaxPool2d, AdaptiveAvgPool2d

##################################################
#Pytorch nn dropout, linear classifier layer unit
from torch.nn import Dropout, Linear

###################
#ConvBN layer unit
from Model.Global_Builder.Unit import (ConvBN, Transpose_ConvBN, DWSep_ConvBN, Transpose_DWSep_ConvBN, Group_ConvBN, Transpose_Group_ConvBN)

######################
#ConvBNAct layer unit
from Model.Global_Builder.Unit import (ConvBNRelu, Transpose_ConvBNRelu, DWSep_ConvBNRelu, Transpose_DWSep_ConvBNRelu, Group_ConvBNRelu, Transpose_Group_ConvBNRelu, \
                                       ConvBNRelu6, Transpose_ConvBNRelu6, DWSep_ConvBNRelu6, Transpose_DWSep_ConvBNRelu6, Group_ConvBNRelu6, Transpose_Group_ConvBNRelu6, \
                                       ConvBNLeakyRelu, Transpose_ConvBNLeakyRelu, DWSep_ConvBNLeakyRelu, Transpose_DWSep_ConvBNLeakyRelu, Group_ConvBNLeakyRelu, Transpose_Group_ConvBNLeakyRelu, \
                                       ConvBNHSwish, Transpose_ConvBNHSwish)

#######################################
#Rest nn functions, typing format list
from Model.Global_Builder.Unit import (ReLU, ReLU6, \
                                       LeakyReLU, Sigmoid, \
                                       H_Sigmoid, H_Swish, \
                                       Conv2d, ConvTranspose2d, \
                                       BatchNorm2d, Any, List, \
                                       Softmax)


#=====================#
# Class Function List #
#=====================#
class Residual_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, stride: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Residual_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stride #####
        self.stride = stride

        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = (chl_in * stride), \
                       kernel_size = 3, stride = stride, padding = 1), \
            ConvBN(chl_in = (chl_in * stride), chl_out = (chl_in * stride), \
                   kernel_size = 3, stride = 1, padding = 1))
        
        ##### Downsample process/flow #####
        if (self.stride == 2):
            self.downsample = ConvBN(chl_in = chl_in, chl_out = (chl_in * stride), \
                                     kernel_size = 1, stride = 2, padding = 0)
        else:
            pass


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------------#
        # Description: Feed-forward pass for residual small (i.e. resnet18, ...) #
        # Input type:                                                            #
        #   - Tensor (input featuremap)                                          #
        # Return type:                                                           #
        #   - Tensor (final featuremap)                                          #
        #------------------------------------------------------------------------#
        stride = self.stride #Stride initialize

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Downsample check #####
        if (stride == 2):
            tmp_identity = self.downsample(tmp_identity)

        else:
            pass

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat
    


class ResidualX_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, stride: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (stride)                          #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResidualX_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stride #####
        self.stride = stride

        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = (chl_in * stride), \
                       kernel_size = 3, stride = stride, padding = 1), \
            Group_ConvBN(chl_in = (chl_in * stride), kernel_size = 3, \
                         stride = 1, padding = 1, groups = groups))
        
        ##### Downsample process/flow #####
        if (self.stride == 2):
            self.downsample = ConvBN(chl_in = chl_in, chl_out = (chl_in * stride), \
                                     kernel_size = 1, stride = 2, padding = 0)
        else:
            pass


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------------------------------#
        # Description: Feed-forward pass for residual-x small (i.e. resnext18, ...) #
        # Input type:                                                               #
        #   - Tensor (input featuremap)                                             # 
        # Return type:                                                              #
        #   - Tensor (final featuremap)                                             #
        #---------------------------------------------------------------------------#
        stride = self.stride #Stride initialize

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Downsample check #####
        if (stride == 2):
            tmp_identity = self.downsample(tmp_identity)

        else:
            pass

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat
    


class DeResidual_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, stride: int = 2) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DeResidual_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            Transpose_ConvBNRelu(chl_in = chl_in, chl_out = chl_in, \
                                 kernel_size = 2, stride = stride, padding = 0), \
            ConvBN(chl_in = chl_in, chl_out = chl_out, \
                   kernel_size = 3, stride = 1, padding = 1))
        
        ##### Upsample process/flow #####
        self.upsample = Transpose_ConvBN(chl_in = chl_in, chl_out = chl_out, \
                                         kernel_size = 2, stride = stride, padding = 0)
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------------------#
        # Description: Feed-forward pass for de-residual small (i.e. de-resnet18, ...) #
        # Input type:                                                                  #
        #   - Tensor (input featuremap)                                                #
        # Return type:                                                                 #
        #   - Tensor (final featuremap)                                                #
        #------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Upsample mapping #####
        tmp_identity = self.upsample(tmp_identity)

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat
    


class DeResidualX_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, groups: int,  \
                 stride: int = 2) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (number of groups)                #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DeResidualX_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            Transpose_ConvBNRelu(chl_in = chl_in, chl_out = chl_in, \
                                 kernel_size = 2, stride = stride, padding = 0), \
            Group_ConvBN(chl_in = chl_in, kernel_size = 3, stride = 1, \
                         padding = 1, groups = groups))
        
        ##### Upsample process/flow #####
        self.upsample = Transpose_ConvBN(chl_in = chl_in, chl_out = chl_out, \
                                         kernel_size = 2, stride = stride, padding = 0)
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------------------------------------#
        # Description: Feed-forward pass for de-residual-x small (i.e. de-resnext18, ...) #
        # Input type:                                                                     #
        #   - Tensor (input featuremap)                                                   #
        # Return type:                                                                    #
        #   - Tensor (final featuremap)                                                   #
        #---------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Upsample mapping #####
        tmp_identity = self.upsample(tmp_identity)

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat



class Residual_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, stride: int, \
                 expand_ratio: int = 4) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (stride)                          #
        #   - int (channel expand ratio)            #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Residual_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stride #####
        self.stride = stride

        ##### Relu #####
        self.relu = ReLU(inplace = True)

        ##### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = chl_out, \
                       kernel_size = 1, stride = 1, padding = 0), \
            ConvBNRelu(chl_in = chl_out, chl_out = chl_out, \
                       kernel_size = 3, stride = stride, padding = 1), \
            ConvBN(chl_in = chl_out, chl_out = (chl_out * expand_ratio), \
                   kernel_size = 1, stride = 1, padding = 0))
        
        ##### Downsample process/flow #####
        if (self.stride == 2):
            self.downsample = ConvBN(chl_in = chl_in, chl_out = (chl_out * expand_ratio), \
                                     kernel_size = 1, stride = 2, padding = 0)
        else:
            pass
            

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------------#
        # Description: Feed-forward pass for residual large (i.e. resnet50, ...) #
        # Input type:                                                            #
        #   - Tensor (input featuremap)                                          #
        # Return type:                                                           #
        #   - Tensor (final featuremap)                                          #
        #------------------------------------------------------------------------#
        stride = self.stride #Stride initialize

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Downsample check #####
        if (stride == 2):
            tmp_identity = self.downsample(tmp_identity)

        else:
            pass

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat
    


class ResidualX_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, stride: int, \
                 groups: int, expand_ratio: int = 4) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (stride)                          #
        #   - int (number of groups)                #
        #   - int (channel expand ratio)            #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResidualX_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stride #####
        self.stride = stride

        ##### Relu #####
        self.relu = ReLU(inplace = True)

        ##### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = chl_out, \
                       kernel_size = 1, stride = 1, padding = 0), \
            Group_ConvBNRelu(chl_in = chl_out, kernel_size = 3, \
                             stride = stride, padding = 1, groups = groups), \
            ConvBN(chl_in = chl_out, chl_out = (chl_out * expand_ratio), \
                   kernel_size = 1, stride = 1, padding = 0))
        
        ##### Downsample process/flow #####
        if (self.stride == 2):
            self.downsample = ConvBN(chl_in = chl_in, chl_out = (chl_out * expand_ratio), \
                                     kernel_size = 1, stride = 2, padding = 0)
        else:
            pass
            

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------------------------------#
        # Description: Feed-forward pass for residual-x large (i.e. resnext50, ...) #
        # Input type:                                                               #
        #   - Tensor (input featuremap)                                             #
        # Return type:                                                              #
        #   - Tensor (final featuremap)                                             #
        #---------------------------------------------------------------------------#
        stride = self.stride #Stride initialize

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Downsample check #####
        if (stride == 2):
            tmp_identity = self.downsample(tmp_identity)

        else:
            pass

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat



class DeResidual_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, stride: int = 2) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DeResidual_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = chl_in, \
                       kernel_size = 1, stride = 1, padding = 0), \
            Transpose_ConvBNRelu(chl_in = chl_in, chl_out = chl_in, \
                                 kernel_size = 2, stride = stride, padding = 0), \
            ConvBN(chl_in = chl_in, chl_out = chl_out, \
                   kernel_size = 1, stride = 1, padding = 0))
        
        ##### Upsample process/flow #####
        self.upsample = Transpose_ConvBN(chl_in = chl_in, chl_out = chl_out, \
                                         kernel_size = 2, stride = stride, padding = 0)
    

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------------------#
        # Description: Feed-forward pass for de-residual large (i.e. de-resnet50, ...) #
        # Input type:                                                                  #
        #   - Tensor (input featuremap)                                                #
        # Return type:                                                                 #
        #   - Tensor (final featuremap)                                                #
        #------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Upsample mapping #####
        tmp_identity = self.upsample(tmp_identity)

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat
    


class DeResidualX_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, groups: int, \
                 stride: int = 2) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (number of groups)                #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DeResidualX_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Relu #####
        self.relu = ReLU(inplace = True)

        #### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu(chl_in = chl_in, chl_out = chl_in, \
                       kernel_size = 1, stride = 1, padding = 0), \
            Transpose_Group_ConvBNRelu(chl_in = chl_in, kernel_size = 2, \
                                       stride = stride, padding = 0, groups = groups), \
            ConvBN(chl_in = chl_in, chl_out = chl_out, \
                   kernel_size = 1, stride = 1, padding = 0))
        
        ##### Upsample process/flow #####
        self.upsample = Transpose_ConvBN(chl_in = chl_in, chl_out = chl_out, \
                                         kernel_size = 2, stride = stride, padding = 0)
    

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------------------------------------#
        # Description: Feed-forward pass for de-residual-x large (i.e. de-resnext50, ...) #
        # Input type:                                                                     #
        #   - Tensor (input featuremap)                                                   #
        # Return type:                                                                    #
        #   - Tensor (final featuremap)                                                   #
        #---------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 3: Upsample mapping #####
        tmp_identity = self.upsample(tmp_identity)

        ##### Step 4: Skip-add connection #####
        final_feat = (tmp_identity + tmp_feat)
        final_feat = self.relu(final_feat)

        return final_feat



class Inverted_Residual(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, stride: int, \
                 expand_ratio: float) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (stride)                          #
        #   - float (channel expand ratio)          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Inverted_Residual, self).__init__() #Inherit from torch.nn.module basis

        assert (stride in [1, 2]) #Issue addressment

        ############
        #Initialize
        ##### Residual connection flag #####
        self.res_connect_flag = (True if ((stride == 1) and (chl_in == chl_out)) else False)

        ##### Bottleneck process/flow #####
        if (expand_ratio == 1.):
            self.bottleneck = Sequential( \
                DWSep_ConvBNRelu6(chl_in = int(chl_in * expand_ratio), kernel_size = 3, \
                                  stride = stride, padding = 1), \
                ConvBN(chl_in = int(chl_in * expand_ratio), chl_out = chl_out, kernel_size = 1, \
                       stride = 1, padding = 0))
            
        else:
            self.bottleneck = Sequential( \
                ConvBNRelu6(chl_in = chl_in, chl_out = int(chl_in * expand_ratio), kernel_size = 1, \
                            stride = 1, padding = 0), \
                DWSep_ConvBNRelu6(chl_in = int(chl_in * expand_ratio), kernel_size = 3, stride = stride, \
                                  padding = 1), \
                ConvBN(chl_in = int(chl_in * expand_ratio), chl_out = chl_out, kernel_size = 1, \
                       stride = 1, padding = 0))
    

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------#
        # Description: Feed-forward pass for inverted residual #
        # Input type:                                          #
        #   - Tensor (input featuremap)                        #
        # Return type:                                         #
        #   - Tensor (final featuremap)                        #
        #------------------------------------------------------#
        res_connect_flag = self.res_connect_flag #Residual connection flag initialize
        
        ####################
        #Whole process/flow
        ##### Step 1: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        ##### Step 2: Residual skip-add connection check #####
        if (res_connect_flag):
            return (inpt_feat + tmp_feat)
        
        else:
            return tmp_feat
        


class DeInverted_Residual(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, expand_ratio: int, \
                 stride: int = 2) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (channel expand ratio)            #
        #   - int (stride)                          #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DeInverted_Residual, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Bottleneck process/flow #####
        self.bottleneck = Sequential( \
            ConvBNRelu6(chl_in = chl_in, chl_out = int(chl_in * expand_ratio), kernel_size = 1, \
                        stride = 1, padding = 0), \
            Transpose_DWSep_ConvBNRelu6(chl_in = int(chl_in * expand_ratio), kernel_size = 2, \
                                        stride = stride, padding = 0), \
            ConvBN(chl_in = int(chl_in * expand_ratio), chl_out = chl_out, kernel_size = 1, \
                   stride = 1, padding = 0))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------------#
        # Description: Feed-forward pass for de-inverted residual #
        # Input type:                                             #
        #   - Tensor (input featuremap)                           #
        # Return type:                                            #
        #   - Tensor (final featuremap)                           #
        #---------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Bottleneck #####
        tmp_feat = self.bottleneck(inpt_feat)

        return tmp_feat



class SE_CBAM(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, reduce_ratio: int = 16) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (channel reduction ratio)         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(SE_CBAM, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Global average pool #####
        self.avgpool = AdaptiveAvgPool2d((1, 1))

        ##### Squeeze and excite linear #####
        self.fc_linear = Sequential( \
            Linear(in_features = chl_in, out_features = (chl_in // reduce_ratio), \
                   bias = False), \
            ReLU(inplace = True), \
            Linear(in_features = (chl_in // reduce_ratio), out_features = chl_in, \
                   bias = False), \
            Sigmoid())
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #----------------------------------------------------------------------------------#
        # Description: Feed-forward pass for squeeze-excite channel-based attention module #
        # Input type:                                                                      #
        #   - Tensor (input featuremap)                                                    #
        # Return type:                                                                     #
        #   - Tensor (final featuremap)                                                    #
        #----------------------------------------------------------------------------------#
        batch_size, channel, _, _ = inpt_feat.size() #Tensor size/shape initialize

        ####################
        #Whole process/flow
        ##### Step 1: Global average pool #####
        tmp_feat = (self.avgpool(inpt_feat)).view(batch_size, channel)

        ##### Step 2: Squeeze and excite #####
        tmp_feat = (self.fc_linear(tmp_feat)).view(batch_size, channel, 1, 1)

        return (inpt_feat * tmp_feat.expand_as(inpt_feat))
    


class ESE_CBAM(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ESE_CBAM, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Global average pool #####
        self.avgpool = AdaptiveAvgPool2d((1, 1))

        ##### Squeeze and excite convolution #####
        self.fc_conv = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = 1, \
                   padding = 0), \
            Sigmoid())
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------------------------------------------------#
        # Description: Feed-forward pass for efficient squeeze-excite channel-based attention module #
        # Input type:                                                                                #
        #   - Tensor (input featuremap)                                                              #
        # Return type:                                                                               #
        #   - Tensor (final featuremap)                                                              #
        #--------------------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Global average pool ##### 
        tmp_feat = self.avgpool(inpt_feat)

        ##### Step 2: Excite multiply #####
        tmp_feat = self.fc_conv(tmp_feat)

        return (inpt_feat * tmp_feat)



class Pool_SBAM(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - None (void, no input)                 #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Pool_SBAM, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Post-process #####
        self.post_process = Sequential( \
            Conv2d(in_channels = 2, out_channels = 1, kernel_size = 3, \
                   padding = 1, bias = False), \
            Sigmoid())


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------------------#
        # Description: Feed-forward pass for pool-based spatial-based attention module #
        # Input type:                                                                  #
        #   - Tensor (input featuremap)                                                #
        # Return type:                                                                 #
        #   - Tensor (final featuremap)                                                #
        #------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Dual-pool (i.e. maxpool, avgpool) #####
        tmp_feat_max, _ = torch_max(inpt_feat, dim = 1, keepdim = True)
        tmp_feat_mean = torch_mean(inpt_feat, dim = 1, keepdim = True)

        ##### Step 2: Pool concatenate #####
        tmp_feat_cat = torch_cat([tmp_feat_max, tmp_feat_mean], dim = 1)

        ##### Step 3: Post-process #####
        tmp_feat = self.post_process(tmp_feat_cat)

        return (inpt_feat * tmp_feat)
    


class Patch_SBAM(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Patch_SBAM, self).__init__() #Inherit from torch.nn.module basis

        from torch.nn import AvgPool2d #Temporal import pytorch nn average pool layer unit

        ############
        #Initialize
        ##### Average pool with zero-padding #####
        self.avg_pool = AvgPool2d(kernel_size = 3, stride = 1, padding = 1)

        ##### Post-process #####
        self.post_process = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = 1, \
                   kernel_size = 1, bias = False), \
            Sigmoid())


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------------------------------------#
        # Description: Feed-forward pass for patch-based spatial-based attention module #
        # Input type:                                                                   #
        #   - Tensor (input featuremap)                                                 #
        # Return type:                                                                  #
        #   - Tensor (final featuremap)                                                 #
        #-------------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Patch-based average pool with zero-padding #####
        tmp_feat = self.avg_pool(inpt_feat)

        ##### Step 2: Post-process #####
        tmp_feat = self.post_process(tmp_feat)

        return (inpt_feat * tmp_feat)
    


class ScaleUp_Add(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self) -> None:
        #--------------------------------------------#
        # Description: Constructor initialize/setup  #
        # Input type:                                #
        #   - None (void, no input)                  #
        # Return type:                               #
        #   - None (void, no return)                 #
        #--------------------------------------------#
        super(ScaleUp_Add, self).__init__() #Inherit from torch.nn.module basis


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, lowlevel_feat: Tensor) -> Tensor:
        #--------------------------------------------------------#
        # Description: Feed-forward pass for scale/resize-up add #
        # Input type:                                            #
        #   - Tensor (input featuremap)                          #
        #   - Tensor (input lower level featuremap)              #
        # Return type:                                           #
        #   - Tensor (final featuremap)                          #
        #--------------------------------------------------------#
        from torch.nn.functional import interpolate #Temporal import pytorch functional interpolation/resize

        ####################
        #Whole process/flow
        ##### Step 1: Scale/Resize-up and add #####
        tmp_feat = interpolate(inpt_feat, scale_factor = 2, mode = 'bilinear', \
                               align_corners = True)
        
        return (tmp_feat + lowlevel_feat)
    


class OSA_V2(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_inner: int, chl_out: int, \
                 times_per_block: int, downsample: bool = False) -> None:
        #--------------------------------------------#
        # Description: Constructor initialize/setup  #
        # Input type:                                #
        #   - int (input channel number)             #
        #   - int (input inner channel number)       #
        #   - int (output channel number)            #
        #   - int (repeat times per block)           #
        #   - bool (downsample flag)                 #
        # Return type:                               #
        #   - None (void, no return)                 #
        #--------------------------------------------#
        super(OSA_V2, self).__init__() #Inherit from torch.nn.module basis

        from torch.nn import ModuleList #Temporal import pytorch nn module list

        ############
        #Initialize
        ##### Downsample flag #####
        self.downsample = downsample

        ##### Downsample process/flow #####
        if (self.downsample):
            self.reduction = ConvBNRelu(chl_in = chl_in, chl_out = chl_in, kernel_size = 3, \
                                        stride = 2, padding = 1)
        else:
            pass

        ##### Layers per-block #####
        self.layers = ModuleList([ConvBNRelu(chl_in = chl_in, chl_out = chl_inner, \
                                  kernel_size = 3, stride = 1, padding = 1)])
        
        for _ in range(1, times_per_block):
            self.layers.append(ConvBNRelu(chl_in = chl_inner, chl_out = chl_inner, \
                                          kernel_size = 3, stride = 1, padding = 1))
            
        ##### Excite pre-convolutions match #####
        self.excite_conv_match = ConvBNRelu(chl_in = (chl_in + (times_per_block * \
                                            chl_inner)), chl_out = chl_out, kernel_size = 1, \
                                            stride = 1, padding = 0)
        
        ##### Efficient squeeze-excitations #####
        self.ese = ESE_CBAM(chl_in = chl_out)


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #----------------------------------------------------------------------------#
        # Description: Feed-forward pass for one-shot aggregation V2 (i.e. VoVNetV2) #
        # Input type:                                                                #
        #   - Tensor (input featuremap)                                              #
        # Return type:                                                               #
        #   - Tensor (final featuremap)                                              #
        #----------------------------------------------------------------------------#
        downsample = self.downsample #Downsample flag initialize

        ####################
        #Whole process/flow
        ##### Step 1: Temporal identity #####
        tmp_identity = inpt_feat

        ##### Step 2: Downsample check #####
        if (downsample):
            tmp_feat = self.reduction(inpt_feat)

        else:
            tmp_feat = inpt_feat

        ##### Step 3: One-shot aggregation #####
        tmp_features = [tmp_feat]

        for layer in self.layers:
            tmp_feat = layer(tmp_feat)

            tmp_features.append(tmp_feat)

        ##### Step 4: Channel concatenation #####
        final_feat = torch_cat(tmp_features, dim = 1)

        ##### Step 5: Pre-convoluions match #####
        final_feat = self.excite_conv_match(final_feat)

        ##### Step 6: Efficient squeeze-excitations #####
        final_feat = self.ese(final_feat)

        ##### Step 7: Residual skip-add connection check #####
        if (not downsample):
            final_feat = (final_feat + tmp_identity)

        else:
            pass

        return final_feat



class IncludeTop_Classifier(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output classes number)           #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(IncludeTop_Classifier, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Global average pool #####
        self.avgpool = AdaptiveAvgPool2d((1, 1))

        ##### Linear classifier #####
        self.fc_linear = Linear(in_features = chl_in, out_features = num_classes)


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------------------#
        # Description: Feed-forward pass for include-top classifier #
        # Input type:                                               #
        #   - Tensor (input featuremap)                             #
        # Return type:                                              #
        #   - Tensor (final featuremap)                             #
        #-----------------------------------------------------------#
        
        ####################
        #Whole process/flow
        ##### Step 1: Global average pool #####
        tmp_feat = self.avgpool(inpt_feat)

        ##### Step 2: Flatten and classified #####
        tmp_feat = tmp_feat.view(tmp_feat.size(0), -1)
        
        final_feat = self.fc_linear(tmp_feat)

        return final_feat