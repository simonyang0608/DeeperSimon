#*************************************************************************************#
# Source: VoVNet.py                                                                   #
#                                                                                     #
# Description: Customized vovnet backbone functions to build whole model architecture #
#                                                                                     #
# Author: SimonYang                                                                   #
#*************************************************************************************#

#================#
# Import Section #
#================#
############################################################
#Pytorch nn module (i.e. basic inherit), sequential, tensor
from Model.Global_Builder.Module import (Module, Sequential, Tensor)

###############################################
#VoVNet block/module series (i.e. V1, V2, ...)
from Model.Global_Builder.Module import OSA_V2

##################
#Rest layer units
from Model.Global_Builder.Module import (ConvBNRelu, Conv2d, BatchNorm2d)

###########################
#Pytorch weight initialize
from torch.nn.init import (kaiming_normal_, zeros_, ones_)

####################
#Typing format list
from Model.Global_Builder.Module import List


#=====================#
# Class Function List #
#=====================#
class VoVNet_V2(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, define_blocks: List[List[int]]) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - List[List[int]] (self-defined blocks) #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(VoVNet_V2, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Stage 1 ~ 5 process/flow #####
        self.stage_1 = Sequential( \
            ConvBNRelu(chl_in = input_channel, chl_out = 32, kernel_size = 3, \
                       stride = 2, padding = 1), \
            ConvBNRelu(chl_in = 32, chl_out = 32, kernel_size = 3, stride = 1, \
                       padding = 1), \
            ConvBNRelu(chl_in = 32, chl_out = 32, kernel_size = 3, stride = 1, \
                       padding = 1)) #Stage 1
        
        self.stage_2 = self.make_stage(define_block = define_blocks[0]) #Stage 2

        self.stage_3 = self.make_stage(define_block = define_blocks[1]) #Stage 3

        self.stage_4 = self.make_stage(define_block = define_blocks[2]) #Stage 4

        self.stage_5 = self.make_stage(define_block = define_blocks[3]) #Stage 5

        ##### Weights/Biases #####
        self.init_weights()


    ########################
    # Member Function List #
    ########################
    def make_stage(self, define_block: List[int]) -> Sequential:
        #-----------------------------------------------#
        # Description: Make stage for stage 1 ~ stage 5 #
        # Input type:                                   #
        #   - List[int] (self-defined block)            #
        # Return type:                                  #
        #   - Sequential (sequential procedure)         #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### Blocks #####
        blocks = [OSA_V2(chl_in = define_block[0], chl_inner = define_block[1], \
                         chl_out = define_block[2], times_per_block = define_block[3], \
                         downsample = True)]
        
        for _ in range(1, define_block[4]):
            blocks.append(OSA_V2(chl_in = define_block[2], chl_inner = define_block[1], \
                                 chl_out = define_block[2], times_per_block = define_block[3]))
            
        return Sequential(*blocks)
    

    def init_weights(self) -> None:
        #-------------------------------------------------------------------------#
        # Description: Weight initialization for vovnetv2 (i.e. vovnetv2_19, ...) #
        # Input type:                                                             #
        #   - None (void, no input)                                               #
        # Return type:                                                            #
        #   - None (void, no return)                                              #    
        #-------------------------------------------------------------------------#

        ############
        #Initialize
        for m in self.modules():

            ##### Convolutions #####
            if (isinstance(m, Conv2d)):
                kaiming_normal_(m.weight)

                if (m.bias is not None):
                    zeros_(m.bias)
                else:
                    pass

            ##### Normalizations #####
            elif isinstance(m, BatchNorm2d):
                ones_(m.weight)
                zeros_(m.bias)

            ##### Others #####
            else:
                pass


    def forward(self, inpt_feat: Tensor) -> List[Tensor]:
        #---------------------------------------------------------------------#
        # Description: Feed-forward pass for vovnetv2 (i.e. vovnetv2_19, ...) #
        # Input type:                                                         #
        #   - Tensor (input featuremap)                                       #
        # Return type:                                                        #
        #   - List[Tensor] (stage featuremaps list)                           #
        #---------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Stage 1 ~ 5 features generated and hooked #####
        tmp_feat_s1 = self.stage_1(inpt_feat) #Stage 1

        tmp_feat_s2 = self.stage_2(tmp_feat_s1) #Stage 2

        tmp_feat_s3 = self.stage_3(tmp_feat_s2) #Stage 3

        tmp_feat_s4 = self.stage_4(tmp_feat_s3) #Stage 4

        tmp_feat_s5 = self.stage_5(tmp_feat_s4) #Stage 5

        return [tmp_feat_s1, tmp_feat_s2, tmp_feat_s3, \
                tmp_feat_s4, tmp_feat_s5]