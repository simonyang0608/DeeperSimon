#*************************************************************************************#
# Source: ResNet.py                                                                   #
#                                                                                     #
# Description: Customized resnet backbone functions to build whole model architecture #
#                                                                                     #
# Author: SimonYang                                                                   #
#*************************************************************************************#

#================#
# Import Section #
#================#
############################################################
#Pytorch nn module (i.e. basic inherit), sequential, tensor
from Model.Global_Builder.Module import (Module, Sequential, Tensor)

#####################################################
#ResNet block/module series (i.e. small, large, ...)
from Model.Global_Builder.Module import Residual_Small, Residual_Large

##################
#Rest layer units
from Model.Global_Builder.Module import (ConvBNRelu, Conv2d, BatchNorm2d)

###########################
#Pytorch weight initialize
from torch.nn.init import (kaiming_normal_, zeros_, constant_)

####################
#Typing format list
from Model.Global_Builder.Module import List


#=====================#
# Class Function List #
#=====================#
class ResNet_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, define_block: List[int]) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - List[int] (self-defined block)        #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResNet_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stage 1 ~ 5 process/flow #####
        self.stage_1 = ConvBNRelu(chl_in = input_channel, chl_out = 32, kernel_size = 3, \
                                  stride = 2, padding = 1) #Stage 1
        
        self.stage_2_orig = ConvBNRelu(chl_in = 32, chl_out = 64, kernel_size = 3, \
                                       stride = 2, padding = 1)
        self.stage_2 = self.make_stage(chl_in = 64, rest_times = define_block[0], \
                                       stride = 1) #Stage 2
        
        self.stage_3 = self.make_stage(chl_in = 64, rest_times = define_block[1], \
                                       stride = 2) #Stage 3
        
        self.stage_4 = self.make_stage(chl_in = 128, rest_times = define_block[2], \
                                       stride = 2) #Stage 4
        
        self.stage_5 = self.make_stage(chl_in = 256, rest_times = define_block[3], \
                                       stride = 2) #Stage 5
        
        ##### Weights/Biases #####
        self.init_weights()
        

    ########################
    # Member Function List #
    ########################
    def make_stage(self, chl_in: int, rest_times: int, stride: int) -> Sequential:
        #-----------------------------------------------#
        # Description: Make stage for stage 1 ~ stage 5 #
        # Input type:                                   #
        #   - int (input channel number)                #
        #   - int (rest times in stage)                 #
        #   - int (stride)                              #
        # Return type:                                  #
        #   - Sequential (sequential procedure)         #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### Blocks #####
        blocks = [Residual_Small(chl_in = chl_in, stride = stride)]

        for _ in range(1, rest_times):
            blocks.append(Residual_Small(chl_in = (chl_in * stride), stride = 1))

        return Sequential(*blocks)
    

    def init_weights(self) -> None:
        #--------------------------------------------------------------------------#
        # Description: Weight initialization for resnet small (i.e. resnet18, ...) #
        # Input type:                                                              #
        #   - None (void, no input)                                                #
        # Return type:                                                             #
        #   - None (void, no return)                                               #
        #--------------------------------------------------------------------------#

        ############
        #Initialize
        for m in self.modules():

            ##### Convolutions #####
            if (isinstance(m, Conv2d)):
                kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

                if (m.bias is not None):
                    zeros_(m.bias)
                else:
                    pass

            ##### Normalizations #####
            elif (isinstance(m, BatchNorm2d)):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

            ##### Others #####
            else:
                pass
    

    def forward(self, inpt_feat: Tensor) -> List[Tensor]:
        #----------------------------------------------------------------------#
        # Description: Feed-forward pass for resnet small (i.e. resnet18, ...) #
        # Input type:                                                          #
        #   - Tensor (input featuremap)                                        #
        # Return type:                                                         #
        #   - List[Tensor] (stage featuremaps list)                            #
        #----------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Stage 1 ~ 5 features generated and hooked #####
        tmp_feat_s1 = self.stage_1(inpt_feat) #Stage 1

        tmp_feat_s2_orig = self.stage_2_orig(tmp_feat_s1)
        tmp_feat_s2 = self.stage_2(tmp_feat_s2_orig) #Stage 2

        tmp_feat_s3 = self.stage_3(tmp_feat_s2) #Stage 3

        tmp_feat_s4 = self.stage_4(tmp_feat_s3) #Stage 4

        tmp_feat_s5 = self.stage_5(tmp_feat_s4) #Stage 5

        return [tmp_feat_s1, tmp_feat_s2, tmp_feat_s3, \
                tmp_feat_s4, tmp_feat_s5]
    


class ResNet_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, define_block: List[int], \
                 expand_ratio: int = 4) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - List[int] (self-defined block)        #
        #   - int (channel expand ratio)            #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResNet_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stage 1 ~ 5 process/flow #####
        self.stage_1 = ConvBNRelu(chl_in = input_channel, chl_out = 32, kernel_size = 3, \
                                  stride = 2, padding = 1) #Stage 1
        
        self.stage_2 = self.make_stage(chl_in = 32, chl_out = 64, rest_times = (define_block[0] + 1), \
                                       stride = 2, expand_ratio = expand_ratio) #Stage 2
        
        self.stage_3 = self.make_stage(chl_in = 256, chl_out = 128, rest_times = define_block[1], \
                                       stride = 2, expand_ratio = expand_ratio) #Stage 3
        
        self.stage_4 = self.make_stage(chl_in = 512, chl_out = 256, rest_times = define_block[2], \
                                       stride = 2, expand_ratio = expand_ratio) #Stage 4
        
        self.stage_5 = self.make_stage(chl_in = 1024, chl_out = 512, rest_times = define_block[3], \
                                       stride = 2, expand_ratio = expand_ratio) #Stage 5
        
        ##### Weights/Biases #####
        self.init_weights()


    ########################
    # Member Function List #
    ########################
    def make_stage(self, chl_in: int, chl_out: int, rest_times: int, \
                   stride: int, expand_ratio: int) -> Sequential:
        #-----------------------------------------------#
        # Description: Make stage for stage 1 ~ stage 5 #
        # Input type:                                   #
        #   - int (input channel number)                #
        #   - int (output channel number)               #
        #   - int (rest times in stage)                 #
        #   - int (stride)                              #
        #   - int (channel expand ratio)                #
        # Return type:                                  #
        #   - Sequential (sequential procedure)         #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### Blocks #####
        blocks = [Residual_Large(chl_in = chl_in, chl_out = chl_out, \
                                 stride = stride)]
        
        for _ in range(1, rest_times):
            blocks.append(Residual_Large(chl_in = (chl_out * expand_ratio), \
                                         chl_out = chl_out, stride = 1))
            
        return Sequential(*blocks)
    

    def init_weights(self) -> None:
        #--------------------------------------------------------------------------#
        # Description: Weight initialization for resnet large (i.e. resnet50, ...) #
        # Input type:                                                              #
        #   - None (void, no input)                                                #
        # Return type:                                                             #
        #   - None (void, no return)                                               #   
        #--------------------------------------------------------------------------#

        ############
        #Initialize
        for m in self.modules():

            ##### Convolutions #####
            if (isinstance(m, Conv2d)):
                kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

                if (m.bias is not None):
                    zeros_(m.bias)
                else:
                    pass

            ##### Normalizations #####
            elif (isinstance(m, BatchNorm2d)):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

            ##### Others #####
            else:
                pass
    

    def forward(self, inpt_feat: Tensor) -> List[Tensor]:
        #----------------------------------------------------------------------#
        # Description: Feed-forward pass for resnet large (i.e. resnet50, ...) #
        # Input type:                                                          #
        #   - Tensor (input featuremap)                                        #
        # Return type:                                                         #
        #   - List[Tensor] (stage featuremaps list)                            #
        #----------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Stage 1 ~ 5 features generate and hooked #####
        tmp_feat_s1 = self.stage_1(inpt_feat) #Stage 1

        tmp_feat_s2 = self.stage_2(tmp_feat_s1) #Stage 2

        tmp_feat_s3 = self.stage_3(tmp_feat_s2) #Stage 3

        tmp_feat_s4 = self.stage_4(tmp_feat_s3) #Stage 4

        tmp_feat_s5 = self.stage_5(tmp_feat_s4) #Stage 5

        return [tmp_feat_s1, tmp_feat_s2, tmp_feat_s3, \
                tmp_feat_s4, tmp_feat_s5]