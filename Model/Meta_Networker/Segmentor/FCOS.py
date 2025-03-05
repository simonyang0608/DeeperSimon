#***********************************************************************************#
# Source: FCOS.py                                                                   #
#                                                                                   #
# Description: Fully convolutional one-stage (i.e. FCOS) model network/architecture #
#                                                                                   #
# Author: SimonYang                                                                 #
#***********************************************************************************#

#================#
# Import Section #
#================#
################################################
#Pytorch nn module (i.e. basic inherit), tensor
from Model.Global_Builder.Module import Module, Tensor

####################
#Backbone functions
from Model.Backbone.General_Mapper import Backbone_Mapper, Stage_Channel_Mapper

###################
#Decoder functions
from Model.Decoder.FPN import FPN, Predict_Channel_Mapper

##############################################
#Rest layer units, modules, nn/rest functions
from Model.Global_Builder.Module import (Conv2d, Patch_SBAM, \
                                         Sigmoid)


#===========================#
# FCOS network/architecture #
#===========================#
class FCOS(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, backbone: str, num_classes: int, \
                 export_flag: bool = False) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - str (backbone)                        #
        #   - int (number of labeled classes)       #
        #   - bool (export flag)                    #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(FCOS, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Number of classes #####
        self.num_classes = num_classes

        ##### Export flag #####
        self.export_flag = export_flag

        ##### Backbone #####
        self.backbone = Backbone_Mapper(input_channel = input_channel, backbone = backbone)

        ##### FPN #####
        self.fpn = FPN(chl_s2 = (Stage_Channel_Mapper(backbone))[1], \
                       chl_s3 = (Stage_Channel_Mapper(backbone))[2], \
                       chl_s4 = (Stage_Channel_Mapper(backbone))[3], \
                       chl_s5 = (Stage_Channel_Mapper(backbone))[4], \
                       chl_out_match = 64)
        
        ##### Smooth process #####
        self.smooth_p2 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 64))[1], \
                                out_channels = 64, kernel_size = 3, stride = 1, \
                                padding = 1)
        
        ##### Patch-based SBAM #####
        self.sbam_p2 = Patch_SBAM(chl_in = 64)
        
        ##### Convolution regressions #####
        self.conv_regression = Conv2d(in_channels = 64, out_channels = ((num_classes + 1) \
                                      if (num_classes > 1) else num_classes), \
                                      kernel_size = 3, stride = 1, padding = 1)
        
        ##### Sigmoid #####
        self.sigmoid = Sigmoid()


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-----------------------------------------------#
        # Description: Feed-forward pass for fcos model #
        # Input type:                                   #
        #   - Tensor (input featuremap)                 #
        # Return type:                                  #
        #   - Tensor (final featuremap)                 #
        #-----------------------------------------------#
        from torch.nn.functional import interpolate #Temporal import pytorch functional interpolation/resize

        ############
        #Initialize
        ##### Number of classes #####
        num_classes = self.num_classes

        ##### Export flag #####
        export_flag = self.export_flag


        ####################
        #Whole process/flow
        ##### Step 1: Backbone #####
        tmp_stage_feat = self.backbone(inpt_feat) #Stage featuremaps

        ##### Step 2: FPN #####
        inpt_feat_s1, inpt_feat_s2, inpt_feat_s3, \
            inpt_feat_s4, inpt_feat_s5 = tmp_stage_feat #Stage featuremaps list
        
        tmp_predict_feat = self.fpn(inpt_feat_s1, inpt_feat_s2, \
                                    inpt_feat_s3, inpt_feat_s4, \
                                    inpt_feat_s5) #Predict featuremaps

        ##### Step 3: Smooth process #####
        tmp_predict_feat[1] = self.smooth_p2(tmp_predict_feat[1]) #P2 updated

        ##### Step 4: Patch-based SBAM #####
        tmp_predict_feat[1] = self.sbam_p2(tmp_predict_feat[1]) #P2 updated

        ##### Step 5: Convolution regressions #####
        tmp_predict_feat[1] = self.conv_regression(tmp_predict_feat[1]) #P2 updated

        ##### Step 6: Scale-up resize #####
        _, inpt_feat_p2, _, _, _ = tmp_predict_feat #Predict featuremaps list updated

        final_feat = interpolate(inpt_feat_p2, scale_factor = 4, mode = 'bilinear', \
                                 align_corners = True) #Final scale-up resize/interpolate
        

        ##### Step 7: Check if number of classes > 1 (i.e. sigmoid or softmax) #####
        if (num_classes > 1):
            from Model.Global_Builder.Module import torch_argmax #Temporal import pytorch argument-max

            ##### Check if it's in export mode or not #####
            if (export_flag):
                return torch_argmax(final_feat, dim = 1)

            return final_feat, torch_argmax(final_feat, dim = 1)
        
        else:
            final_feat = self.sigmoid(final_feat)
        
            return final_feat