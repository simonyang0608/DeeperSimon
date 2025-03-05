#**************************************************************#
# Source: Unit.py                                              #
#                                                              #
# Description: Customized layer functions to build model block #
#                                                              #
# Author: SimonYang                                            #
#**************************************************************#

#================#
# Import Section #
#================#
############################################################
#Pytorch nn module (i.e. basic inherit), sequential, tensor
from Model.Global_Builder.Activation import Module, Tensor
from torch.nn import Sequential

##################################################################################
#Pytorch nn convolution/transpose convolution (i.e. conv2d, convtranspose2d, ...)
from torch.nn import Conv2d, ConvTranspose2d

#########################################
#Pytorch nn normalization (i.e. BN, ...)
from torch.nn import BatchNorm2d

######################
#Activation functions
from Model.Global_Builder.Activation import (ReLU, ReLU6, \
                                             LeakyReLU, Sigmoid, \
                                             H_Sigmoid, H_Swish, \
                                             Softmax)

####################
#Typing format list
from typing import Any, List


#=====================#
# Class Function List #
#=====================#
class ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                   stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------#
        # Description: Feed-forward pass for conv+bn #
        # Input type:                                #
        #   - Tensor (input featuremap)              #
        # Return type:                               #
        #   - Tensor (final featuremap)              #
        #--------------------------------------------#
        opr = self.opr #Conv+bn operation initialize

        return opr(inpt_feat)
    


class ConvBNHSwish(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ConvBNHSwish, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                   stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            H_Swish())
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #---------------------------------------------------#
        # Description: Feed-forward pass for conv+bn+hswish #
        # Input type:                                       #
        #   - Tensor (input featuremap)                     #
        # Return type:                                      #
        #   - Tensor (final featuremap)                     #
        #---------------------------------------------------#
        opr = self.opr #Conv+bn+hswish operation initialize

        return opr(inpt_feat)
            

  
class ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                   stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------#
        # Description: Feed-forward pass for conv+bn+relu6 #
        # Input type:                                      #
        #   - Tensor (input featuremap)                    #
        # Return type:                                     #
        #   - Tensor (final featuremap)                    #
        #--------------------------------------------------#
        opr = self.opr #Conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                   stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------#
        # Description: Feed-forward pass for conv+bn+relu #
        # Input type:                                     #
        #   - Tensor (input featuremap)                   #
        # Return type:                                    #
        #   - Tensor (final featuremap)                   #
        #-------------------------------------------------#
        opr = self.opr #Conv+bn+relu operation initialize

        return opr(inpt_feat)



class ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                   stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------#
        # Description: Feed-forward pass for conv+bn+leakyrelu #
        # Input type:                                          #
        #   - Tensor (input featuremap)                        #
        # Return type:                                         #
        #   - Tensor (final featuremap)                        #
        #------------------------------------------------------#
        opr = self.opr #Conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)
    


class Transpose_ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                            stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------#
        # Description: Feed-forward pass for transpose_conv+bn #
        # Input type:                                          #
        #   - Tensor (input featuremap)                        #
        # Return type:                                         #
        #   - Tensor (final featuremap)                        #
        #------------------------------------------------------#
        opr = self.opr #Transpose_conv+bn operation initialize

        return opr(inpt_feat)
    


class Transpose_ConvBNHSwish(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_ConvBNHSwish, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                            stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            H_Swish())
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_conv+bn+hswish #
        # Input type:                                                 #
        #   - Tensor (input featuremap)                               #
        # Return type:                                                #
        #   - Tensor (final featuremap)                               #
        #-------------------------------------------------------------#
        opr = self.opr #Transpose_conv+bn+hswish operation initialize

        return opr(inpt_feat)
  


class Transpose_ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                            stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_conv+bn+relu6 #
        # Input type:                                                #
        #   - Tensor (input featuremap)                              #
        # Return type:                                               #
        #   - Tensor (final featuremap)                              #
        #------------------------------------------------------------#
        opr = self.opr #Transpose_conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class Transpose_ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                            stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------------------#
        # Description: Feed-forward pass for transpose_conv+bn+relu #
        # Input type:                                               #
        #   - Tensor (input featuremap)                             #
        # Return type:                                              #
        #   - Tensor (final featuremap)                             #
        #-----------------------------------------------------------#
        opr = self.opr #Transpose_conv+bn+relu operation initialize

        return opr(inpt_feat)



class Transpose_ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, chl_out: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (output channel number)           #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_out, kernel_size = kernel_size, \
                            stride = stride, padding = padding, bias = False), \
            BatchNorm2d(num_features = chl_out), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #----------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_conv+bn+leakyrelu #
        # Input type:                                                    #
        #   - Tensor (input featuremap)                                  #
        # Return type:                                                   #
        #   - Tensor (final featuremap)                                  #
        #----------------------------------------------------------------#
        opr = self.opr #Transpose_conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)
    


class DWSep_ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DWSep_ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------#
        # Description: Feed-forward pass for dwsep conv+bn #
        # Input type:                                      #
        #   - Tensor (input featuremap)                    #
        # Return type:                                     #
        #   - Tensor (final featuremap)                    #
        #--------------------------------------------------#
        opr = self.opr #DWSep conv+bn operation initialize

        return opr(inpt_feat)
  


class DWSep_ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DWSep_ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------------#
        # Description: Feed-forward pass for dwsep conv+bn+relu6 #
        # Input type:                                            #
        #   - Tensor (input featuremap)                          #
        # Return type:                                           #
        #   - Tensor (final featuremap)                          #
        #--------------------------------------------------------#
        opr = self.opr #DWSep conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class DWSep_ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DWSep_ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------------#
        # Description: Feed-forward pass for dwsep conv+bn+relu #
        # Input type:                                           #
        #   - Tensor (input featuremap)                         #
        # Return type:                                          #
        #   - Tensor (final featuremap)                         #
        #-------------------------------------------------------#
        opr = self.opr #DWSep conv+bn+relu operation initialize

        return opr(inpt_feat)



class DWSep_ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DWSep_ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for dwsep conv+bn+leakyrelu #
        # Input type:                                                #
        #   - Tensor (input featuremap)                              #
        # Return type:                                               #
        #   - Tensor (final featuremap)                              #
        #------------------------------------------------------------#
        opr = self.opr #DWSep conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)
    


class Transpose_DWSep_ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_DWSep_ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_dwsep conv+bn #
        # Input type:                                                #
        #   - Tensor (input featuremap)                              #
        # Return type:                                               #
        #   - Tensor (final featuremap)                              #
        #------------------------------------------------------------#
        opr = self.opr #Transpose_dwsep conv+bn operation initialize

        return opr(inpt_feat)
  


class Transpose_DWSep_ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_DWSep_ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_dwsep conv+bn+relu6 #
        # Input type:                                                      #
        #   - Tensor (input featuremap)                                    #
        # Return type:                                                     #
        #   - Tensor (final featuremap)                                    #
        #------------------------------------------------------------------#
        opr = self.opr #Transpose_dwsep conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class Transpose_DWSep_ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_DWSep_ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_dwsep conv+bn+relu #
        # Input type:                                                     #
        #   - Tensor (input featuremap)                                   #
        # Return type:                                                    #
        #   - Tensor (final featuremap)                                   #
        #-----------------------------------------------------------------#
        opr = self.opr #Transpose_dwsep conv+bn+relu operation initialize

        return opr(inpt_feat)



class Transpose_DWSep_ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_DWSep_ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = chl_in, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #----------------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_dwsep conv+bn+leakyrelu #
        # Input type:                                                          #
        #   - Tensor (input featuremap)                                        #
        # Return type:                                                         #
        #   - Tensor (final featuremap)                                        #
        #----------------------------------------------------------------------#
        opr = self.opr #Transpose_dwsep conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)



class Group_ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Group_ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------#
        # Description: Feed-forward pass for group conv+bn #
        # Input type:                                      #
        #   - Tensor (input featuremap)                    #
        # Return type:                                     #
        #   - Tensor (final featuremap)                    #
        #--------------------------------------------------#
        opr = self.opr #Group conv+bn operation initialize

        return opr(inpt_feat)
  


class Group_ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Group_ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------------#
        # Description: Feed-forward pass for group conv+bn+relu6 #
        # Input type:                                            #
        #   - Tensor (input featuremap)                          #
        # Return type:                                           #
        #   - Tensor (final featuremap)                          #
        #--------------------------------------------------------#
        opr = self.opr #Group conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class Group_ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Group_ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------------#
        # Description: Feed-forward pass for group conv+bn+relu #
        # Input type:                                           #
        #   - Tensor (input featuremap)                         #
        # Return type:                                          #
        #   - Tensor (final featuremap)                         #
        #-------------------------------------------------------#
        opr = self.opr #Group conv+bn+relu operation initialize

        return opr(inpt_feat)



class Group_ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Group_ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            Conv2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                   stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for group conv+bn+leakyrelu #
        # Input type:                                                #
        #   - Tensor (input featuremap)                              #
        # Return type:                                               #
        #   - Tensor (final featuremap)                              #
        #------------------------------------------------------------#
        opr = self.opr #Group conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)



class Transpose_Group_ConvBN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_Group_ConvBN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_group conv+bn #
        # Input type:                                                #
        #   - Tensor (input featuremap)                              #
        # Return type:                                               #
        #   - Tensor (final featuremap)                              #
        #------------------------------------------------------------#
        opr = self.opr #Transpose_group conv+bn operation initialize

        return opr(inpt_feat)
  


class Transpose_Group_ConvBNRelu6(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_Group_ConvBNRelu6, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU6(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #------------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_group conv+bn+relu6 #
        # Input type:                                                      #
        #   - Tensor (input featuremap)                                    #
        # Return type:                                                     #
        #   - Tensor (final featuremap)                                    #
        #------------------------------------------------------------------#
        opr = self.opr #Transpose_group conv+bn+relu6 operation initialize

        return opr(inpt_feat)



class Transpose_Group_ConvBNRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_Group_ConvBNRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            ReLU(inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_group conv+bn+relu #
        # Input type:                                                     #
        #   - Tensor (input featuremap)                                   #
        # Return type:                                                    #
        #   - Tensor (final featuremap)                                   #
        #-----------------------------------------------------------------#
        opr = self.opr #Transpose_group conv+bn+relu operation initialize

        return opr(inpt_feat)



class Transpose_Group_ConvBNLeakyRelu(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_in: int, kernel_size: int, \
                 stride: int, padding: int, groups: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel number)            #
        #   - int (kernel size)                     #
        #   - int (stride)                          #
        #   - int (padding)                         #
        #   - int (number of groups)                #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Transpose_Group_ConvBNLeakyRelu, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Main/Major operation #####
        self.opr = Sequential( \
            ConvTranspose2d(in_channels = chl_in, out_channels = chl_in, kernel_size = kernel_size, \
                            stride = stride, padding = padding, groups = groups, bias = False), \
            BatchNorm2d(num_features = chl_in), \
            LeakyReLU(negative_slope = 0.1, inplace = True))
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #----------------------------------------------------------------------#
        # Description: Feed-forward pass for transpose_group conv+bn+leakyrelu #
        # Input type:                                                          #
        #   - Tensor (input featuremap)                                        #
        # Return type:                                                         #
        #   - Tensor (final featuremap)                                        #
        #----------------------------------------------------------------------#
        opr = self.opr #Transpose_group conv+bn+leakyrelu operation initialize

        return opr(inpt_feat)



