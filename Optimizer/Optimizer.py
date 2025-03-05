#***************************************************************************#
# Source: Optimizer.py                                                      #
#                                                                           #
# Description: Customized optimizer functions for model training/validation #
#                                                                           #
# Author: SimonYang                                                         #
#***************************************************************************#

#================#
# Import Section #
#================#
###################################################
#Pytorch optimizer, nn module (i.e. basic inherit)
from torch.optim import Optimizer
from torch.nn import Module

#############################
#Pytorch optimizer functions
from torch.optim import (SGD, RMSprop, Adagrad, \
                         Adadelta, Adam, Adamax, \
                         AdamW)


#=================#
# Mapper Function #
#=================#
def Optimizer_Mapper(optimizer: str, model: Module,  lr: float, \
                     momentum: float, weight_decay: float) -> Optimizer:
    #-----------------------------------------------------------#
    # Description: Customized mapper for self-defined optimizer #
    # Input type:                                               #
    #   - str (self-defined optimizer)                          #
    #   - Module (self-defined model)                           #
    #   - float (learning rate)                                 #
    #   - float (momentum)                                      #
    #   - float (weight decay)                                  #
    # Return type:                                              #
    #   - Optimizer (result optimizer)                          #
    #-----------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ##########################################
    #Mapper process with different optimizers
    ##### SGD #####
    mapper_dict['sgd'] = SGD(params = model.parameters(), lr = lr, \
                             momentum = momentum, weight_decay = weight_decay)
    ##### RMS-Prop #####
    mapper_dict['rmsprop'] = RMSprop(params = model.parameters(), lr = lr, \
                                     momentum = momentum, weight_decay = weight_decay)
    ##### Ada #####
    mapper_dict['adagrad'] = Adagrad(params = model.parameters(), lr = lr, \
                                     weight_decay = weight_decay)
    mapper_dict['adadelta'] = Adadelta(params = model.parameters(), lr = lr, \
                                       weight_decay = weight_decay)
    
    ##### Adam #####
    mapper_dict['adam'] = Adam(params = model.parameters(), lr = lr, \
                               weight_decay = weight_decay)
    mapper_dict['adamw'] = AdamW(params = model.parameters(), lr = lr, \
                                 weight_decay = weight_decay)
    mapper_dict['adamax'] = Adamax(params = model.parameters(), lr = lr, \
                                   weight_decay = weight_decay)
    
    return mapper_dict[optimizer]