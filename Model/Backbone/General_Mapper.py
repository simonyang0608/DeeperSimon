#****************************************************#
# Source: General_Mapper.py                          #
#                                                    #
# Description: General mapper for backbone functions #
#                                                    #
# Author: SimonYang                                  #
#****************************************************#

#================#
# Import Section #
#================#
########################################
#Pytorch nn module (i.e. basic inherit)
from Model.Global_Builder.Module import Module

####################
#typing format list
from Model.Global_Builder.Module import List

#################
#Backbone series
##### ResNet #####
from Model.Backbone.ResNet import ResNet_Small, ResNet_Large

##### ResNeXt #####
from Model.Backbone.ResNeXt import ResNeXt_Small, ResNeXt_Large

##### VoVNet #####
from Model.Backbone.VoVNet import VoVNet_V2


#=================#
# Mapper Function #
#=================#
def Backbone_Mapper(input_channel: int, backbone: str) -> Module:
    #----------------------------------------------------------#
    # Description: Customized mapper for self-defined backbone #
    # Input type:                                              #
    #   - int (input channel)                                  #
    #   - str (self-defined backbone)                          #
    # Return type:                                             #
    #   - Module (result backbone)                             #
    #----------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ###############################################
    #Mapper process with different backbone series
    ##### ResNet #####
    if ('resnet' in backbone):
        mapper_dict['resnet18'] = ResNet_Small(input_channel = input_channel, \
                                               define_block = [2, 2, 2, 2])
        mapper_dict['resnet34'] = ResNet_Small(input_channel = input_channel, \
                                               define_block = [3, 4, 6, 3])
        mapper_dict['resnet50'] = ResNet_Large(input_channel = input_channel, \
                                               define_block = [3, 4, 6, 3])
        mapper_dict['resnet101'] = ResNet_Large(input_channel = input_channel, \
                                                define_block = [3, 4, 23, 3])
        mapper_dict['resnet152'] = ResNet_Large(input_channel = input_channel, \
                                                define_block = [3, 8, 36, 3])
        
    ##### ResNeXt #####
    elif ('resnext' in backbone):
        mapper_dict['resnext18'] = ResNeXt_Small(input_channel = input_channel, \
                                                 define_block = [2, 2, 2, 2])
        mapper_dict['resnext34'] = ResNeXt_Small(input_channel = input_channel, \
                                                 define_block = [3, 4, 6, 3])
        mapper_dict['resnext50'] = ResNeXt_Large(input_channel = input_channel, \
                                                 define_block = [3, 4, 6, 3])
        mapper_dict['resnext101'] = ResNeXt_Large(input_channel = input_channel, \
                                                  define_block = [3, 4, 23, 3])
        mapper_dict['resnext152'] = ResNeXt_Large(input_channel = input_channel, \
                                                  define_block = [3, 8, 36, 3])
        
    ##### VoVNet #####
    elif ('vovnet' in backbone):
        mapper_dict['vovnetv2_19'] = VoVNet_V2(input_channel = input_channel, \
                                               define_blocks = [[32, 32, 64, 3, 2], \
                                                                [64, 48, 128, 3, 2], \
                                                                [128, 64, 256, 3, 2], \
                                                                [256, 80, 348, 3, 2]])
        mapper_dict['vovnetv2_27'] = VoVNet_V2(input_channel = input_channel, \
                                               define_blocks = [[32, 32, 128, 5, 2], \
                                                                [128, 48, 256, 5, 2], \
                                                                [256, 64, 348, 5, 2], \
                                                                [348, 80, 512, 5, 2]])
        mapper_dict['vovnetv2_39'] = VoVNet_V2(input_channel = input_channel, \
                                               define_blocks = [[32, 32, 256, 5, 2], \
                                                                [256, 48, 512, 5, 2], \
                                                                [512, 64, 768, 5, 2], \
                                                                [768, 80, 1024, 5, 2]])
        mapper_dict['vovnetv2_57'] = VoVNet_V2(input_channel = input_channel, \
                                               define_blocks = [[32, 64, 256, 5, 2], \
                                                                [256, 80, 512, 5, 2], \
                                                                [512, 96, 768, 5, 4], \
                                                                [768, 112, 1024, 5, 3]])
        mapper_dict['vovnetv2_99'] = VoVNet_V2(input_channel = input_channel, \
                                               define_blocks = [[32, 64, 256, 5, 2], \
                                                                [256, 80, 512, 5, 3], \
                                                                [512, 96, 768, 5, 9], \
                                                                [768, 112, 1024, 5, 3]])
    
    return mapper_dict[backbone]



def Stage_Channel_Mapper(backbone: str) -> List[int]:
    #---------------------------------------------------------------#
    # Description: Customized mapper for stage channel numbers list #
    # Input type:                                                   #
    #   - str (self-defined backbone)                               #
    # Return type:                                                  #
    #   - List[int] (result stage channel list)                     #
    #---------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ###############################################
    #Mapper process with different backbone series
    ##### ResNet #####
    if ('resnet' in backbone):
        mapper_dict['resnet18'] = [32, 64, 128, 256, 512]
        mapper_dict['resnet34'] = [32, 64, 128, 256, 512]
        mapper_dict['resnet50'] = [32, 256, 512, 1024, 2048]
        mapper_dict['resnet101'] = [32, 256, 512, 1024, 2048]
        mapper_dict['resnet152'] = [32, 256, 512, 1024, 2048]

    ##### ResNeXt #####
    elif ('resnext' in backbone):
        mapper_dict['resnext18'] = [32, 64, 128, 256, 512]
        mapper_dict['resnext34'] = [32, 64, 128, 256, 512]
        mapper_dict['resnext50'] = [32, 256, 512, 1024, 2048]
        mapper_dict['resnext101'] = [32, 256, 512, 1024, 2048]
        mapper_dict['resnext152'] = [32, 256, 512, 1024, 2048]

    ##### VoVNet #####
    elif ('vovnet' in backbone):
        mapper_dict['vovnetv2_19'] = [32, 64, 128, 256, 348]
        mapper_dict['vovnetv2_27'] = [32, 128, 256, 348, 512]
        mapper_dict['vovnetv2_39'] = [32, 256, 512, 768, 1024]
        mapper_dict['vovnetv2_57'] = [32, 256, 512, 768, 1024]
        mapper_dict['vovnetv2_99'] = [32, 256, 512, 768, 1024]
    
    return mapper_dict[backbone]