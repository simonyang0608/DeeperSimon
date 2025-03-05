#*************************************************************#
# Source: General_Mapper.py                                   #
#                                                             #
# Description: General mapper for models with different tasks #
#                                                             #
# Author: SimonYang                                           #
#*************************************************************#

#================#
# Import Section #
#================#
########################################
#Pytorch nn module (i.e. basic inherit)
from Model.Global_Builder.Module import Module

############################
#Meta networks/models tasks
##### Segmentation #####
from Model.Meta_Networker.Segmentor.MemSeg import MemSeg
from Model.Meta_Networker.Segmentor.FCOS import FCOS

##### Detection #####
from Model.Meta_Networker.Detector.Faster_RCNN import Faster_RCNN


#=================#
# Mapper Function #
#=================#
def Model_Mapper(cfg: dict, num_classes: int) -> Module:
    #----------------------------------------------------------------------------#
    # Description: Customized mapper for self-defined model network/architecture #
    # Input type:                                                                #
    #   - dict (whole config information)                                        #
    #   - int (number of labeled classes)                                        #
    # Return type:                                                               #
    #   - Module (result model)                                                  #
    #----------------------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    #############################################
    #Mapper process with different meta networks
    ##### MeMSeg #####
    mapper_dict['memseg'] = MemSeg(input_channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                                   backbone = cfg['MODEL']['BACKBONE'], \
                                   num_classes = num_classes, \
                                   export_flag = cfg['export'])
    ##### FCOS #####
    mapper_dict['fcos'] = FCOS(input_channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                               backbone = cfg['MODEL']['BACKBONE'], \
                               num_classes = num_classes, \
                               export_flag = cfg['export'])
    ##### Faster-RCNN #####
    mapper_dict['faster_rcnn'] = Faster_RCNN(input_channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                                             backbone = cfg['MODEL']['BACKBONE'], \
                                             num_classes = num_classes, \
                                             roi_pool_size = cfg['INPUT']['ROI_POOL_SIZE'], \
                                             min_size = cfg['INPUT']['RESIZE_SCALE']['MIN_SIZE'], \
                                             max_size = cfg['INPUT']['RESIZE_SCALE']['MAX_SIZE'])

    return mapper_dict[cfg['MODEL']['META_NETWORK']]