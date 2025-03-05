#***************************************************************************************#
# Source: UNet.py                                                                       #
#                                                                                       #
# Description: Customized U-network decoder functions to build whole model architecture #
#                                                                                       #
# Author: SimonYang                                                                     #
#***************************************************************************************#

#================#
# Import Section #
#================#
#############
#Pytorch cat
from Model.Global_Builder.Module import torch_cat

############################################################
#Pytorch nn module (i.e. basic inherit), sequential, tensor
from Model.Global_Builder.Module import (Module, Sequential, Tensor)

##############################
#U-network based block/module
from Model.Global_Builder.Module import (DeResidual_Small, DeResidual_Large, \
                                         DeResidualX_Small, DeResidualX_Large)

##################
#Rest layer units
from Model.Global_Builder.Module import ConvTranspose2d, ESE_CBAM


#=====================#
# Class Function List #
#=====================#
class ResUNet_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input predict 1 channel number)  #
        #   - int (input predict 2 channel number)  #
        #   - int (input predict 3 channel number)  #
        #   - int (input predict 4 channel number)  #
        #   - int (input predict 5 channel number)  #
        #   - int (number of labeled classes)       #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResUNet_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Level 1 ~ 5 process/flow #####
        self.level_1 = ConvTranspose2d(in_channels = (chl_p1 * 2), out_channels = num_classes, \
                                       kernel_size = 2, stride = 2, padding = 0) #Level 1
        
        self.level_2 = DeResidual_Small(chl_in = (chl_p2 * 2), chl_out = chl_p1) #Level 2
        
        self.level_3 = DeResidual_Small(chl_in = (chl_p3 * 2), chl_out = chl_p2) #Level 3

        self.level_4 = DeResidual_Small(chl_in = (chl_p4 * 2), chl_out = chl_p3) #Level 4

        self.level_5 = DeResidual_Small(chl_in = chl_p5, chl_out = chl_p4) #Level 5
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_p1: Tensor, inpt_feat_p2: Tensor, \
                inpt_feat_p3: Tensor, inpt_feat_p4: Tensor, \
                inpt_feat_p5: Tensor) -> Tensor:
        #------------------------------------------------------------------------#
        # Description: Feed-forward pass for resunet small (i.e. resunet18, ...) #
        # Input type:                                                            #
        #   - Tensor (input predict 1 featuremaps)                               #
        #   - Tensor (input predict 2 featuremaps)                               #
        #   - Tensor (input predict 3 featuremaps)                               #
        #   - Tensor (input predict 4 featuremaps)                               #
        #   - Tensor (input predict 5 featuremaps)                               #
        # Return type:                                                           #
        #   - Tensor (level 1 (i.e. final level) featuremaps)                    #
        #------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Level 1 ~ 5 features generate and hooked #####
        tmp_feat_l5 = self.level_5(inpt_feat_p5) #Level 5

        tmp_catfeat_54 = torch_cat([tmp_feat_l5, inpt_feat_p4], dim = 1)
        tmp_feat_l4 = self.level_4(tmp_catfeat_54) #Level 4

        tmp_catfeat_43 = torch_cat([tmp_feat_l4, inpt_feat_p3], dim = 1)
        tmp_feat_l3 = self.level_3(tmp_catfeat_43) #Level 3

        tmp_catfeat_32 = torch_cat([tmp_feat_l3, inpt_feat_p2], dim = 1)
        tmp_feat_l2 = self.level_2(tmp_catfeat_32) #Level 2

        tmp_catfeat_21 = torch_cat([tmp_feat_l2, inpt_feat_p1], dim = 1)
        tmp_feat_l1 = self.level_1(tmp_catfeat_21) #Level 1

        return tmp_feat_l1
    


class ResUNeXt_Small(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input predict 1 channel number)  #
        #   - int (input predict 2 channel number)  #
        #   - int (input predict 3 channel number)  #
        #   - int (input predict 4 channel number)  #
        #   - int (input predict 5 channel number)  #
        #   - int (number of labeled classes)       #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResUNeXt_Small, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Level 1 ~ 5 process/flow #####
        self.level_1 = ConvTranspose2d(in_channels = (chl_p1 * 2), out_channels = num_classes, \
                                       kernel_size = 2, stride = 2, padding = 0) #Level 1
        
        self.level_2 = DeResidualX_Small(chl_in = (chl_p2 * 2), chl_out = chl_p1, groups = 8) #Level 2
        
        self.level_3 = DeResidualX_Small(chl_in = (chl_p3 * 2), chl_out = chl_p2, groups = 8) #Level 3

        self.level_4 = DeResidualX_Small(chl_in = (chl_p4 * 2), chl_out = chl_p3, groups = 8) #Level 4

        self.level_5 = DeResidualX_Small(chl_in = chl_p5, chl_out = chl_p4, groups = 8) #Level 5
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_p1: Tensor, inpt_feat_p2: Tensor, \
                inpt_feat_p3: Tensor, inpt_feat_p4: Tensor, \
                inpt_feat_p5: Tensor) -> Tensor:
        #--------------------------------------------------------------------------#
        # Description: Feed-forward pass for resunext small (i.e. resunext18, ...) #
        # Input type:                                                              #
        #   - Tensor (input predict 1 featuremaps)                                 #
        #   - Tensor (input predict 2 featuremaps)                                 #
        #   - Tensor (input predict 3 featuremaps)                                 #
        #   - Tensor (input predict 4 featuremaps)                                 #
        #   - Tensor (input predict 5 featuremaps)                                 #
        # Return type:                                                             #
        #   - Tensor (level 1 (i.e. final level) featuremaps)                      #
        #--------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Level 1 ~ 5 features generate and hooked #####
        tmp_feat_l5 = self.level_5(inpt_feat_p5) #Level 5

        tmp_catfeat_54 = torch_cat([tmp_feat_l5, inpt_feat_p4], dim = 1)
        tmp_feat_l4 = self.level_4(tmp_catfeat_54) #Level 4

        tmp_catfeat_43 = torch_cat([tmp_feat_l4, inpt_feat_p3], dim = 1)
        tmp_feat_l3 = self.level_3(tmp_catfeat_43) #Level 3

        tmp_catfeat_32 = torch_cat([tmp_feat_l3, inpt_feat_p2], dim = 1)
        tmp_feat_l2 = self.level_2(tmp_catfeat_32) #Level 2

        tmp_catfeat_21 = torch_cat([tmp_feat_l2, inpt_feat_p1], dim = 1)
        tmp_feat_l1 = self.level_1(tmp_catfeat_21) #Level 1

        return tmp_feat_l1
    


class ResUNet_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input predict 1 channel number)  #
        #   - int (input predict 2 channel number)  #
        #   - int (input predict 3 channel number)  #
        #   - int (input predict 4 channel number)  #
        #   - int (input predict 5 channel number)  #
        #   - int (number of labeled classes)       #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResUNet_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Level 1 ~ 5 process/flow #####
        self.level_1 = ConvTranspose2d(in_channels = (chl_p1 * 2), out_channels = num_classes, \
                                       kernel_size = 2, stride = 2, padding = 0) #Level 1
        
        self.level_2 = DeResidual_Large(chl_in = (chl_p2 * 2), chl_out = chl_p1) #Level 2
        
        self.level_3 = DeResidual_Large(chl_in = (chl_p3 * 2), chl_out = chl_p2) #Level 3
        
        self.level_4 = DeResidual_Large(chl_in = (chl_p4 * 2), chl_out = chl_p3) #Level 4
        
        self.level_5 = DeResidual_Large(chl_in = chl_p5, chl_out = chl_p4) #Level 5


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_p1: Tensor, inpt_feat_p2: Tensor, \
                inpt_feat_p3: Tensor, inpt_feat_p4: Tensor, \
                inpt_feat_p5: Tensor) -> Tensor:
        #------------------------------------------------------------------------#
        # Description: Feed-forward pass for resunet large (i.e. resunet50, ...) #
        # Input type:                                                            #
        #   - Tensor (input predict 1 featuremaps)                               #
        #   - Tensor (input predict 2 featuremaps)                               #
        #   - Tensor (input predict 3 featuremaps)                               #
        #   - Tensor (input predict 4 featuremaps)                               #
        #   - Tensor (input predict 5 featuremaps)                               #
        # Return type:                                                           #
        #   - Tensor (level 1 (i.e. final level) featuremaps)                    #
        #------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Level 1 ~ 5 features generate and hooked #####
        tmp_feat_l5 = self.level_5(inpt_feat_p5) #Level 5

        tmp_catfeat_54 = torch_cat([tmp_feat_l5, inpt_feat_p4], dim = 1)
        tmp_feat_l4 = self.level_4(tmp_catfeat_54) #Level 4

        tmp_catfeat_43 = torch_cat([tmp_feat_l4, inpt_feat_p3], dim = 1)
        tmp_feat_l3 = self.level_3(tmp_catfeat_43) #Level 3

        tmp_catfeat_32 = torch_cat([tmp_feat_l3, inpt_feat_p2], dim = 1)
        tmp_feat_l2 = self.level_2(tmp_catfeat_32) #Level 2

        tmp_catfeat_21 = torch_cat([tmp_feat_l2, inpt_feat_p1], dim = 1)
        tmp_feat_l1 = self.level_1(tmp_catfeat_21) #Level 1

        return tmp_feat_l1
    


class ResUNeXt_Large(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input predict 1 channel number)  #
        #   - int (input predict 2 channel number)  #
        #   - int (input predict 3 channel number)  #
        #   - int (input predict 4 channel number)  #
        #   - int (input predict 5 channel number)  #
        #   - int (number of labeled classes)       #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(ResUNeXt_Large, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Level 1 ~ 5 process/flow #####
        self.level_1 = ConvTranspose2d(in_channels = (chl_p1 * 2), out_channels = num_classes, \
                                       kernel_size = 2, stride = 2, padding = 0) #Level 1
        
        self.level_2 = DeResidualX_Large(chl_in = (chl_p2 * 2), chl_out = chl_p1, groups = 8) #Level 2
        
        self.level_3 = DeResidualX_Large(chl_in = (chl_p3 * 2), chl_out = chl_p2, groups = 8) #Level 3
        
        self.level_4 = DeResidualX_Large(chl_in = (chl_p4 * 2), chl_out = chl_p3, groups = 8) #Level 4
        
        self.level_5 = DeResidualX_Large(chl_in = chl_p5, chl_out = chl_p4, groups = 8) #Level 5


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_p1: Tensor, inpt_feat_p2: Tensor, \
                inpt_feat_p3: Tensor, inpt_feat_p4: Tensor, \
                inpt_feat_p5: Tensor) -> Tensor:
        #--------------------------------------------------------------------------#
        # Description: Feed-forward pass for resunext large (i.e. resunext50, ...) #
        # Input type:                                                              #
        #   - Tensor (input predict 1 featuremaps)                                 #
        #   - Tensor (input predict 2 featuremaps)                                 #
        #   - Tensor (input predict 3 featuremaps)                                 #
        #   - Tensor (input predict 4 featuremaps)                                 #
        #   - Tensor (input predict 5 featuremaps)                                 #
        # Return type:                                                             #
        #   - Tensor (level 1 (i.e. final level) featuremaps)                      #
        #--------------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Level 1 ~ 5 features generate and hooked #####
        tmp_feat_l5 = self.level_5(inpt_feat_p5) #Level 5

        tmp_catfeat_54 = torch_cat([tmp_feat_l5, inpt_feat_p4], dim = 1)
        tmp_feat_l4 = self.level_4(tmp_catfeat_54) #Level 4

        tmp_catfeat_43 = torch_cat([tmp_feat_l4, inpt_feat_p3], dim = 1)
        tmp_feat_l3 = self.level_3(tmp_catfeat_43) #Level 3

        tmp_catfeat_32 = torch_cat([tmp_feat_l3, inpt_feat_p2], dim = 1)
        tmp_feat_l2 = self.level_2(tmp_catfeat_32) #Level 2

        tmp_catfeat_21 = torch_cat([tmp_feat_l2, inpt_feat_p1], dim = 1)
        tmp_feat_l1 = self.level_1(tmp_catfeat_21) #Level 1

        return tmp_feat_l1
    


class VoVUNet_V2(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - int (input predict 1 channel number)  #
        #   - int (input predict 2 channel number)  #
        #   - int (input predict 3 channel number)  #
        #   - int (input predict 4 channel number)  #
        #   - int (input predict 5 channel number)  #
        #   - int (number of labeled classes)       #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(VoVUNet_V2, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Level 1 ~ 5 process/flow #####
        self.level_1 = ConvTranspose2d(in_channels = (chl_p1 * 2), out_channels = num_classes, \
                                       kernel_size = 2, stride = 2, padding = 0) #Level 1
        
        self.level_2 = Sequential( \
            DeResidual_Small(chl_in = (chl_p2 * 2), chl_out = chl_p1), \
            ESE_CBAM(chl_in = chl_p1)) #Level 2
        
        self.level_3 = Sequential( \
            DeResidual_Small(chl_in = (chl_p3 * 2), chl_out = chl_p2), \
            ESE_CBAM(chl_in = chl_p2)) #Level 3
        
        self.level_4 = Sequential( \
            DeResidual_Small(chl_in = (chl_p4 * 2), chl_out = chl_p3), \
            ESE_CBAM(chl_in = chl_p3)) #Level 4
        
        self.level_5 = Sequential( \
            DeResidual_Small(chl_in = chl_p5, chl_out = chl_p4), \
            ESE_CBAM(chl_in = chl_p4)) #Level 5
        

    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat_p1: Tensor, inpt_feat_p2: Tensor, \
                inpt_feat_p3: Tensor, inpt_feat_p4: Tensor, \
                inpt_feat_p5: Tensor) -> Tensor:
        #-----------------------------------------------------------------------#
        # Description: Feed-forward pass for vovunetv2 (i.e. vovunetv2_19, ...) #
        # Input type:                                                           #
        #   - Tensor (input predict 1 featuremaps)                              #
        #   - Tensor (input predict 2 featuremaps)                              #
        #   - Tensor (input predict 3 featuremaps)                              #
        #   - Tensor (input predict 4 featuremaps)                              #
        #   - Tensor (input predict 5 featuremaps)                              #
        # Return type:                                                          #
        #   - Tensor (level 1 (i.e. final level) featuremaps)                   #
        #-----------------------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Level 1 ~ 5 features generate and hooked #####
        tmp_feat_l5 = self.level_5(inpt_feat_p5) #Level 5

        tmp_catfeat_54 = torch_cat([tmp_feat_l5, inpt_feat_p4], dim = 1)
        tmp_feat_l4 = self.level_4(tmp_catfeat_54) #Level 4

        tmp_catfeat_43 = torch_cat([tmp_feat_l4, inpt_feat_p3], dim = 1)
        tmp_feat_l3 = self.level_3(tmp_catfeat_43) #Level 3

        tmp_catfeat_32 = torch_cat([tmp_feat_l3, inpt_feat_p2], dim = 1)
        tmp_feat_l2 = self.level_2(tmp_catfeat_32) #Level 2

        tmp_catfeat_21 = torch_cat([tmp_feat_l2, inpt_feat_p1], dim = 1)
        tmp_feat_l1 = self.level_1(tmp_catfeat_21) #Level 1

        return tmp_feat_l1



#=================#
# Mapper Function #
#=================#
def UNet_Mapper(backbone: str, chl_p1: int, chl_p2: int, chl_p3: int, \
                 chl_p4: int, chl_p5: int, num_classes: int) -> Module:
    #--------------------------------------------------------------#
    # Description: Customized mapper for self-defined unet decoder #
    # Input type:                                                  #
    #   - str (self-defined backbone)                              #
    #   - int (input predict 1 channel number)                     #
    #   - int (input predict 2 channel number)                     #
    #   - int (input predict 3 channel number)                     #
    #   - int (input predict 4 channel number)                     #
    #   - int (input predict 5 channel number)                     #
    #   - int (number of labeled classes)                          #
    # Return type:                                                 #
    #   - Module (result unet decoder)                             #
    #--------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ###############################################
    #Mapper process with different backbone series
    ##### ResNet #####
    if ('resnet' in backbone):
        mapper_dict['resnet18'] = ResUNet_Small(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnet34'] = ResUNet_Small(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnet50'] = ResUNet_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnet101'] = ResUNet_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                 chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                 chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnet152'] = ResUNet_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                 chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                 chl_p5 = chl_p5, num_classes = num_classes)
        
    ##### ResNeXt #####
    elif ('resnext' in backbone):
        mapper_dict['resnext18'] = ResUNeXt_Small(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                  chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                  chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnext34'] = ResUNeXt_Small(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                  chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                  chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnext50'] = ResUNeXt_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                  chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                  chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnext101'] = ResUNeXt_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                   chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                   chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['resnext152'] = ResUNeXt_Large(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                   chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                   chl_p5 = chl_p5, num_classes = num_classes)
    
    ##### VoVNet #####
    elif ('vovnet' in backbone):
        mapper_dict['vovnetv2_19'] = VoVUNet_V2(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['vovnetv2_27'] = VoVUNet_V2(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['vovnetv2_39'] = VoVUNet_V2(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['vovnetv2_57'] = VoVUNet_V2(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
        mapper_dict['vovnetv2_99'] = VoVUNet_V2(chl_p1 = chl_p1, chl_p2 = chl_p2, \
                                                chl_p3 = chl_p3, chl_p4 = chl_p4, \
                                                chl_p5 = chl_p5, num_classes = num_classes)
    
    return mapper_dict[backbone]