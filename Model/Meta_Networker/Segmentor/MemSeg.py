#***************************************************************************#
# Source: MemSeg.py                                                         #
#                                                                           #
# Description: Memory segmentation (i.e. Memseg) model network/architecture #
#                                                                           #
# Author: SimonYang                                                         #
#***************************************************************************#

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
from Model.Decoder.UNet import UNet_Mapper

#########################################
#Rest layer units, modules, nn functions
from Model.Global_Builder.Module import (Conv2d, Patch_SBAM, \
                                         Sigmoid)


#=============================#
# Memseg network/architecture #
#=============================#
class MemSeg(Module):

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
        super(MemSeg, self).__init__() #Inherit from torch.nn.module basis

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
                       chl_out_match = 128)
        
        ##### Smooth process #####
        self.smooth_p1 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 128))[0], \
                                out_channels = ((Predict_Channel_Mapper(backbone, chl_out_match = 128))[0] // 2), \
                                kernel_size = 3, stride = 1, padding = 1)
        
        self.smooth_p2 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 128))[1], \
                                out_channels = 48, kernel_size = 3, stride = 1, \
                                padding = 1)
        
        self.smooth_p3 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 128))[2], \
                                out_channels = 64, kernel_size = 3, stride = 1, \
                                padding = 1)
        
        self.smooth_p4 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 128))[3], \
                                out_channels = 128, kernel_size = 3, stride = 1, \
                                padding = 1)
        
        ##### Patch-based SBAM #####
        self.sbam_p2 = Patch_SBAM(chl_in = 48)
        self.sbam_p3 = Patch_SBAM(chl_in = 64)
        self.sbam_p4 = Patch_SBAM(chl_in = 128)
        
        ##### UNet decoder #####
        self.unet_decoder = UNet_Mapper(backbone = backbone, \
                                        chl_p1 = ((Predict_Channel_Mapper(backbone, chl_out_match = 128))[0] // 2), \
                                        chl_p2 = 48, chl_p3 = 64, chl_p4 = 128, \
                                        chl_p5 = (Predict_Channel_Mapper(backbone, chl_out_match = 128))[4], \
                                        num_classes = ((num_classes + 1) if (num_classes > 1) else num_classes))
        
        ##### Sigmoid #####
        self.sigmoid = Sigmoid()


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #-------------------------------------------------#
        # Description: Feed-forward pass for memseg model #
        # Input type:                                     #
        #   - Tensor (input featuremap)                   #
        # Return type:                                    #
        #   - Tensor (final featuremap)                   #
        #-------------------------------------------------#

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
        tmp_predict_feat[0] = self.smooth_p1(tmp_predict_feat[0]) #P1 updated
        tmp_predict_feat[1] = self.smooth_p2(tmp_predict_feat[1]) #P2 updated
        tmp_predict_feat[2] = self.smooth_p3(tmp_predict_feat[2]) #P3 updated
        tmp_predict_feat[3] = self.smooth_p4(tmp_predict_feat[3]) #P4 updated

        ##### Step 4: Patch-based SBAM #####
        tmp_predict_feat[1] = self.sbam_p2(tmp_predict_feat[1]) #P2 updated
        tmp_predict_feat[2] = self.sbam_p3(tmp_predict_feat[2]) #P3 updated
        tmp_predict_feat[3] = self.sbam_p4(tmp_predict_feat[3]) #P4 updated

        ##### Step 5: UNet decoder #####
        inpt_feat_p1, inpt_feat_p2, inpt_feat_p3, \
            inpt_feat_p4, inpt_feat_p5 = tmp_predict_feat #Predict featuremaps list updated

        final_feat = self.unet_decoder(inpt_feat_p1, inpt_feat_p2, \
                                       inpt_feat_p3, inpt_feat_p4, \
                                       inpt_feat_p5) #Final featuremap
        
        
        ##### Step 6: Check if number of classes > 1 (i.e. sigmoid or softmax) #####
        if (num_classes > 1):
            from Model.Global_Builder.Module import torch_argmax #Temporal import pytorch argument-max

            ##### Check if it's in export mode or not #####
            if (export_flag):
                return torch_argmax(final_feat, dim = 1)

            return final_feat, torch_argmax(final_feat, dim = 1)
        
        else:
            final_feat = self.sigmoid(final_feat)
            
            return final_feat