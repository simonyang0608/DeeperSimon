#*******************************************************************#
# Source: Activation.py                                             #
#                                                                   #
# Description: Customized activation functions to build model layer #
#                                                                   #
# Author: SimonYang                                                 #
#*******************************************************************#

#================#
# Import Section #
#================#
################################################
#Pytorch nn module (i.e. basic inherit), tensor
from torch.nn import Module
from torch import Tensor

##################################################
#Pytorch nn activations (i.e. relu, sigmoid, ...)
from torch.nn import (ReLU, ReLU6, LeakyReLU, \
                      Sigmoid, Softmax)


#=====================#
# Class Function List #
#=====================#
class H_Sigmoid(Module):

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
        super(H_Sigmoid, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        self.relu = ReLU6(inplace = True)


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------#
        # Description: Feed-forward pass for hard-sigmoid #
        # Input type:                                     #
        #   - Tensor (input featuremap)                   #
        # Return type:                                    #
        #   - Tensor (final featuremap)                   #
        #-------------------------------------------------#
        relu = self.relu #Relu initialize

        return (relu(inpt_feat + 3) / 6)
    


class H_Swish(Module):

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
        super(H_Swish, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        self.h_sigmoid = H_Sigmoid()


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------#
        # Description: Feed-forward pass for hard-swish #
        # Input type:                                   #
        #   - Tensor (input featuremap)                 #
        # Return type:                                  #
        #   - Tensor (final featuremap)                 #
        #-----------------------------------------------#
        h_sigmoid = self.h_sigmoid #Hard-sigmoid initialize

        return (inpt_feat * h_sigmoid(inpt_feat))
