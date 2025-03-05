#**********************************************************************************************************#
# Source: Faster_RCNN.py                                                                                   #
#                                                                                                          #
# Description: Faster regional convolutional neural network (i.e. Faster R-CNN) model network/architecture #
#                                                                                                          #
# Author: SimonYang                                                                                        #
#**********************************************************************************************************#

#================#
# Import Section #
#================#
################################################
#Pytorch nn module (i.e. basic inherit), tensor
from Model.Global_Builder.Module import Module, Tensor

################################
#Pytorch faster r-cnn functions
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

####################
#Backbone functions
from Model.Backbone.General_Mapper import Backbone_Mapper, Stage_Channel_Mapper

###################
#Decoder functions
from Model.Decoder.FPN import FPN, Predict_Channel_Mapper

##############################################
#Rest layer units, modules, nn/rest functions
from Model.Global_Builder.Module import (Conv2d, Patch_SBAM, ConvBNRelu)

####################
#Typing format list
from Model.Global_Builder.Module import Any, List


#===========================#
# Feature extractor/encoder #
#===========================#
class Feature_Extractor(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, backbone: str) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - str (backbone)                        #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Feature_Extractor, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Backbone #####
        self.backbone = Backbone_Mapper(input_channel = input_channel, backbone = backbone)

        ##### FPN #####
        self.fpn = FPN(chl_s2 = (Stage_Channel_Mapper(backbone))[1], \
                       chl_s3 = (Stage_Channel_Mapper(backbone))[2], \
                       chl_s4 = (Stage_Channel_Mapper(backbone))[3], \
                       chl_s5 = (Stage_Channel_Mapper(backbone))[4], \
                       chl_out_match = 112)
        
        ##### Smooth process #####
        self.smooth_p2 = Conv2d(in_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 112))[1], \
                                out_channels = 112, kernel_size = 3, stride = 1, \
                                padding = 1)
        
        ##### Patch-based SBAM #####
        self.sbam_p2 = Patch_SBAM(chl_in = 112)
        
        ##### Down-sample process #####
        self.downsample_1 = ConvBNRelu(chl_in = 112, chl_out = 112, kernel_size = 3, \
                                       stride = 2, padding = 1)
        
        self.downsample_2 = ConvBNRelu(chl_in = 112, chl_out = 112, kernel_size = 3, \
                                       stride = 2, padding = 1)
        
        self.downsample_3 = ConvBNRelu(chl_in = 112, chl_out = 112, kernel_size = 3, \
                                       stride = 2, padding = 1)
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor) -> Tensor:
        #--------------------------------------------------------------#
        # Description: Feed-forward pass for feature extractor/encoder #
        # Input type:                                                  #
        #   - Tensor (input featuremap)                                #
        # Return type:                                                 #
        #   - Tensor (final featuremap)                                #
        #--------------------------------------------------------------#

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

        ##### Step 5: Down-sample process #####
        _, inpt_feat_p2, _, _, _ = tmp_predict_feat #Predict featuremaps list updated
        
        final_feat = self.downsample_1(inpt_feat_p2) #1st step
        final_feat = self.downsample_2(final_feat) #2nd step
        final_feat = self.downsample_3(final_feat) #3rd step

        return final_feat



#===================================#
# Faster R-CNN network/architecture #
#===================================#
class Faster_RCNN(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, input_channel: int, backbone: str, \
                 num_classes: int, roi_pool_size: int, \
                 min_size: int, max_size: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input channel)                   #
        #   - str (backbone)                        #
        #   - int (number of labeled classes)       #
        #   - int (input roi pool size)             #
        #   - int (input minimun scale/size)        #
        #   - int (input maximun scale/size)        #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Faster_RCNN, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Backbone #####
        self.backbone = Feature_Extractor(input_channel = input_channel, backbone = backbone)
                
        ##### Last output channel stage from backbone #####
        (self.backbone).out_channels = (Predict_Channel_Mapper(backbone, chl_out_match = 112))[1]

        ##### Anchor generator #####
        self.anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512),), \
                                                aspect_ratios = ((0.5, 1.0, 2.0),))
            
        ##### ROI align #####
        self.roi_pooler = MultiScaleRoIAlign(featmap_names = ['0'], \
                                             output_size = roi_pool_size, \
                                             sampling_ratio = 2)
            
        ##### Build whole model architecture #####
        self.whole_model = FasterRCNN(backbone = self.backbone, \
                                      num_classes = (num_classes + 1), \
                                      min_size = min_size, \
                                      max_size = max_size, \
                                      rpn_anchor_generator = self.anchor_generator, \
                                      box_roi_pool = self.roi_pooler)


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target: List[dict]) -> Any:
        #-------------------------------------------------------------#
        # Description: Feed-forward pass for faster r-cnn model       #
        # Input type:                                                 #
        #   - Tensor (input featuremap)                               #
        #   - List[dict] (bounding-boxes, class indexes informations) #
        # Return type:                                                #
        #   - Any (final output features informations)                #
        #-------------------------------------------------------------#
        
        ####################
        #Whole process/flow
        ##### Step 1: Check if it's in train or validate mode #####
        if (self.training):
            outputs = self.whole_model(inpt_feat, target) #Output losses
        else:
            outputs = self.whole_model(inpt_feat) #Output predictions

        return outputs
        
