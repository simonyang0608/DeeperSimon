#*********************************************************************************************#
# Source: FPN.py                                                                              #
#                                                                                             #
# Description: Customized feature-pyramid decoder functions to build whole model architecture #
#                                                                                             #
# Author: SimonYang                                                                           #
#*********************************************************************************************#

#================#
# Import Section #
#================#
################################################
#Pytorch nn module (i.e. basic inherit), tensor
from Model.Global_Builder.Module import Module, Tensor

####################################
#Feature pyramid based block/module
from Model.Global_Builder.Module import ScaleUp_Add

##################
#Rest layer units
from Model.Global_Builder.Module import Conv2d

####################
#Typing format list
from Model.Global_Builder.Module import List


#=====================#
# Class Function List #
#=====================#
class FPN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_s2: int, chl_s3: int, chl_s4: int, \
                 chl_s5: int, chl_out_match: int) -> None:
        #--------------------------------------------#
        # Description: Constructor initialize/setup  #
        # Input type:                                #
        #   - int (input stage 2 channel number)     #
        #   - int (input stage 3 channel number)     #
        #   - int (input stage 4 channel number)     #
        #   - int (input stage 5 channel number)     #
        #   - int (output predict channel match)     #
        # Return type:                               #
        #   - None (void, no return)                 #
        #--------------------------------------------#
        super(FPN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Scale-up add #####
        self.upsample_add = ScaleUp_Add()

        ##### Predict 2 ~ 5 process/flow #####
        self.predict_2 = Conv2d(in_channels = chl_s2, out_channels = chl_out_match, \
                                kernel_size = 1, stride = 1, padding = 0) #Predict 2
        
        self.predict_3 = Conv2d(in_channels = chl_s3, out_channels = chl_out_match, \
                                kernel_size = 1, stride = 1, padding = 0) #Predict 3
        
        self.predict_4 = Conv2d(in_channels = chl_s4, out_channels = chl_out_match, \
                                kernel_size = 1, stride = 1, padding = 0) #predict 4
        
        self.predict_5 = Conv2d(in_channels = chl_s5, out_channels = chl_out_match, \
                                kernel_size = 1, stride = 1, padding = 0) #predict 5


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_s1: Tensor, inpt_feat_s2: Tensor, \
                inpt_feat_s3: Tensor, inpt_feat_s4: Tensor, \
                inpt_feat_s5: Tensor) -> List[Tensor]:
        #------------------------------------------------------------#
        # Description: Feed-forward pass for feature pyramid network #
        # Input type:                                                #
        #   - Tensor (input stage 1 featuremaps)                     #
        #   - Tensor (input stage 2 featuremaps)                     #
        #   - Tensor (input stage 3 featuremaps)                     #
        #   - Tensor (input stage 4 featuremaps)                     #
        #   - Tensor (input stage 5 featuremaps)                     #
        # Return type:                                               #
        #   - List[Tensor] (predict featuremaps list)                #
        #------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Predict 1 ~ 5 features generate and hooked #####
        tmp_feat_p5 = inpt_feat_s5 #Predict 5
        
        tmp_feat_p4 = self.upsample_add(self.predict_5(tmp_feat_p5), \
                                        self.predict_4(inpt_feat_s4)) #Predict 4

        tmp_feat_p3 = self.upsample_add(tmp_feat_p4, self.predict_3(inpt_feat_s3)) #Predict 3

        tmp_feat_p2 = self.upsample_add(tmp_feat_p3, self.predict_2(inpt_feat_s2)) #Predict 2

        tmp_feat_p1 = inpt_feat_s1 #Predict 1

        return [tmp_feat_p1, tmp_feat_p2, tmp_feat_p3, \
                tmp_feat_p4, tmp_feat_p5]



#=================#
# Mapper Function #
#=================#
def Predict_Channel_Mapper(backbone: str, chl_out_match: int) -> List[int]:
    #-----------------------------------------------------------------#
    # Description: Customized mapper for predict channel numbers list #
    # Input type:                                                     #
    #   - str (self-defined backbone)                                 #
    #   - int (output predict channel match)                          #
    # Return type:                                                    #
    #   - List[int] (result predict channel list)                     #
    #-----------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ###############################################
    #Mapper process with different backbone series
    ##### ResNet #####
    if ('resnet' in backbone):
        mapper_dict['resnet18'] = [32, chl_out_match, chl_out_match, \
                                   chl_out_match, 512]
        mapper_dict['resnet34'] = [32, chl_out_match, chl_out_match, \
                                   chl_out_match, 512]
        mapper_dict['resnet50'] = [32, chl_out_match, chl_out_match, \
                                   chl_out_match, 2048]
        mapper_dict['resnet101'] = [32, chl_out_match, chl_out_match, \
                                    chl_out_match, 2048]
        mapper_dict['resnet152'] = [32, chl_out_match, chl_out_match, \
                                    chl_out_match, 2048]
        
    ##### ResNeXt #####
    elif ('resnext' in backbone):
        mapper_dict['resnext18'] = [32, chl_out_match, chl_out_match, \
                                    chl_out_match, 512]
        mapper_dict['resnext34'] = [32, chl_out_match, chl_out_match, \
                                    chl_out_match, 512]
        mapper_dict['resnext50'] = [32, chl_out_match, chl_out_match, \
                                    chl_out_match, 2048]
        mapper_dict['resnext101'] = [32, chl_out_match, chl_out_match, \
                                     chl_out_match, 2048]
        mapper_dict['resnext152'] = [32, chl_out_match, chl_out_match, \
                                     chl_out_match, 2048]

    ##### VoVNet #####
    elif ('vovnet' in backbone):
        mapper_dict['vovnetv2_19'] = [32, chl_out_match, chl_out_match, \
                                      chl_out_match, 348]
        mapper_dict['vovnetv2_27'] = [32, chl_out_match, chl_out_match, \
                                      chl_out_match, 512]
        mapper_dict['vovnetv2_39'] = [32, chl_out_match, chl_out_match, \
                                      chl_out_match, 1024]
        mapper_dict['vovnetv2_57'] = [32, chl_out_match, chl_out_match, \
                                      chl_out_match, 1024]
        mapper_dict['vovnetv2_99'] = [32, chl_out_match, chl_out_match, \
                                      chl_out_match, 1024]
    
    return mapper_dict[backbone]


