#*****************************************************************************************#
# Source: Measurer.py                                                                     #
#                                                                                         #
# Description: Measure/Summarize related information for whole model network/architecture #
#                                                                                         #
# Author: SimonYang                                                                       #
#*****************************************************************************************#

#================#
# Import Section #
#================#
########################################
#Pytorch nn module (i.e. basic inherit)
from Model.Global_Builder.Module import Module

#################
#Pytorch randoms
from torch import randn

#################################
#Pytorch model FLOPS, parameters
from thop import profile

####################
#Typing format list
from Model.Global_Builder.Module import Any


#======================#
# Define Function List #
#======================#
def Summarizer(logging: Any, model: Module, device: Any, \
               channel: int, height: int, width: int, \
               task: str, batch_size: int = 1) -> None:
    #----------------------------------------------------#
    # Description: Summarize for whole model information #
    # Input type:                                        #
    #   - Any (logging record)                           #
    #   - Module (self-defined model)                    #
    #   - Any (gpu/cpu device)                           #
    #   - int (input channel)                            #
    #   - int (input height)                             #
    #   - int (input width)                              #
    #   - str (input tasks type)                         #
    #   - int (input batch-size)                         #
    # Return type:                                       #
    #   - None (void, no return)                         #
    #----------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Whole architecture #####
    logging.info("==> Whole network/architecture: {}".format(model))
    
    ##### Step 2: FLOPs/Parameters #####
    if (task == 'detection'): #Detection
        flops, parameters = profile(model = (model.whole_model), inputs = (randn(batch_size, channel, \
                                    height, width).to(device), None,))
        
    else: #Segmentation/Classification
        flops, parameters = profile(model = model, inputs = (randn(batch_size, channel, \
                                    height, width).to(device),))
    
    logging.info("===> Total FLOPs: {} G".format((flops / (1000 ** 3))))
    logging.info("===> Total parameters: {} M".format((parameters / (1000 ** 2))))