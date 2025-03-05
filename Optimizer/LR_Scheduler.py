#*****************************************************************************************#
# Source: LR_Scheduler.py                                                                 #
#                                                                                         #
# Description: Customized learning rate scheduler functions for model training/validation #
#                                                                                         #
# Author: SimonYang                                                                       #
#*****************************************************************************************#

#================#
# Import Section #
#================#
######################################################
#Pytorch lr scheduler, optimizer (i.e. basic inherit)
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

################################
#Pytorch lr scheduler functions
from torch.optim.lr_scheduler import (StepLR, ExponentialLR, MultiStepLR)

##################
#Bisect functions
from bisect import bisect_right

############################
#Fvcore parameter functions
from fvcore.common.param_scheduler import (CompositeParamScheduler, \
                                           ConstantParamScheduler,
                                           LinearParamScheduler,
                                           MultiStepParamScheduler, \
                                           CosineParamScheduler, \
                                           ParamScheduler)

####################
#Typing format list
from typing import Any, List


#=====================#
# Class Function List #
#=====================#
class WarmupParamScheduler(CompositeParamScheduler):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, scheduler: ParamScheduler, \
                 warmup_factor: float, warmup_length: float, \
                 warmup_method: str = "linear") -> None:
        #----------------------------------------------#
        # Description: Constructor initialize/setup    #
        # Input type:                                  #
        #   - ParamScheduler (scheduler)               #
        #   - float (warmup_factor)                    #
        #   - float (warmup_length)                    #
        #   - str (warmup_method)                      #
        # Return type:                                 #
        #   - None (void, no return)                   #
        #----------------------------------------------#

        ############
        #Initialize
        ##### Value to reach when warmup ends #####
        end_value = scheduler(warmup_length)
        start_value = (warmup_factor * scheduler(0.0))

        if (warmup_method == "constant"):
            warmup = ConstantParamScheduler(start_value)

        elif (warmup_method == "linear"):
            warmup = LinearParamScheduler(start_value, end_value)

        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))


        ##### Additional inherit #####
        super().__init__([warmup, scheduler], interval_scaling = ["rescaled", "fixed"], \
                          lengths = [warmup_length, 1 - warmup_length])



class LRMultiplier(_LRScheduler):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, optimizer: Optimizer, multiplier: ParamScheduler, \
                 max_iter: int, last_iter: int = -1):
        #----------------------------------------------#
        # Description: Constructor initialize/setup    #
        # Input type:                                  #
        #   - Optimizer (self-defined optimizer)       #
        #   - float (warmup_factor)                    #
        #   - float (warmup_length)                    #
        #   - str (warmup_method)                      #
        #   - int (last epochs/iterations)             #
        # Return type:                                 #
        #   - None (void, no return)                   #
        #----------------------------------------------#
        if (not isinstance(multiplier, ParamScheduler)):
            raise ValueError("_LRMultiplier(multiplier=) must be an instance of fvcore " \
                             "ParamScheduler. Got {} instead.".format(multiplier)) #Issue addressment
        else:
            pass

        ############
        #Initialize
        ##### Multiplier/Max iteration #####
        self._multiplier = multiplier
        self._max_iter = max_iter

        ##### Additional inherit #####
        super().__init__(optimizer, last_epoch = last_iter)


    ########################
    # Member Function List #
    ########################
    def state_dict(self) -> dict:
        #----------------------------------------------------------------------------------#
        # Description: Fvcore schedulers are stateless. Only keep pytorch scheduler states #
        # Input type:                                                                      #
        #   - None (void, no input)                                                        # 
        # Return type:                                                                     #
        #   - dict (state dict informations)                                               #
        #----------------------------------------------------------------------------------#
        
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}


    def get_lr(self) -> List[float]:
        #------------------------------------------------#
        # Description: Get/Query learning rate values    #
        # Input type:                                    #
        #   - None (void, no input)                      #  
        # Return type:                                   #
        #   - List[float] (updated learning rate values) #
        #------------------------------------------------#

        ############
        #Initialize
        ##### Multiplier #####
        multiplier = self._multiplier(self.last_epoch / self._max_iter)

        return [(base_lr * multiplier) for base_lr in self.base_lrs]



class WarmupMultiStepLR(_LRScheduler):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, optimizer: Optimizer, milestones: List[int], \
                 gamma: float = 0.1, warmup_factor: float = 0.001, \
                 warmup_iters: int = 1000, warmup_method: str = "linear", \
                 last_epoch: int = -1) -> None:
        #----------------------------------------------#
        # Description: Constructor initialize/setup    #
        # Input type:                                  #
        #   - Optimizer (self-defined optimizer)       #
        #   - List[int] (milestones)                   #
        #   - float (gamma value)                      #
        #   - float (warmup_factor)                    #
        #   - int (warmup_iters)                       #
        #   - str (warmup_method)                      #
        #   - int (last epochs/iterations)             #
        # Return type:                                 #
        #   - None (void, no return)                   #
        #----------------------------------------------#
        if (not (list(milestones) == sorted(milestones))):
            raise ValueError("Milestones should be a list of" 
                             "increasing integers. Got {}".format(milestones)) #Issue addressment
        else:
            pass
        
        ############
        #Initialize
        ##### Milestones #####
        self.milestones = milestones

        ##### Gamma value #####
        self.gamma = gamma

        ##### Warm-up parameters #####
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        ##### Additional inherit #####
        super().__init__(optimizer, last_epoch)


    ########################
    # Member Function List #
    ########################
    def get_lr(self) -> List[float]:
        #------------------------------------------------#
        # Description: Get/Query learning rate values    #
        # Input type:                                    #
        #   - None (void, no input)                      #  
        # Return type:                                   #
        #   - List[float] (updated learning rate values) #
        #------------------------------------------------#

        ############
        #Initialize
        ##### Warm-up factor #####
        warmup_factor = self._get_warmup_factor_at_iter(method = self.warmup_method, \
                                                        iter = self.last_epoch, \
                                                        warmup_iters = self.warmup_iters, \
                                                        warmup_factor = self.warmup_factor)
        
        return [(base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, \
                 self.last_epoch)) for base_lr in self.base_lrs]


    def _compute_values(self) -> List[float]:
        #-----------------------------------------------#
        # Description: Compute learning rate values     #
        # Input type:                                   #
        #   - None (void, no input)                     #  
        # Return type:                                  #
        #   - List[float] (output learning rate values) #
        #-----------------------------------------------#

        return self.get_lr()
    

    def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, \
                                   warmup_factor: float) -> float:
        #--------------------------------------------------------------------#
        # Description: Get/Query warm-up factors at specific epoch/iteration #
        # Input type:                                                        #
        #   - str (self-defined methods)                                     #
        #   - int (epoch/iteration)                                          #
        #   - int (warmup_iters)                                             #
        #   - float (warmup_factor)                                          #
        # Return type:                                                       #
        #   - float (output learning rate factors)                           #
        #--------------------------------------------------------------------#

        #################################
        #Customized self-defined methods
        if (iter >= warmup_iters):
            return 1.0
        else:
            pass
        
        if (method == "constant"):
            return warmup_factor
        
        elif (method == "linear"):
            alpha = (iter / warmup_iters)

            return ((warmup_factor * (1 - alpha)) + alpha)
        
        else:
            raise ValueError("Unknown warmup method: {}".format(method))



class WarmupCosineLR(_LRScheduler):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, optimizer: Optimizer, max_iters: int, \
                 warmup_factor: float = 0.001, warmup_iters: int = 1000, \
                 warmup_method: str = "linear", last_epoch: int = -1) -> None:
        #----------------------------------------------#
        # Description: Constructor initialize/setup    #
        # Input type:                                  #
        #   - Optimizer (self-defined optimizer)       #
        #   - int (max epochs/iterations)              #
        #   - float (warmup_factor)                    #
        #   - int (warmup_iters)                       #
        #   - str (warmup_method)                      #
        #   - int (last epochs/iterations)             #
        # Return type:                                 #
        #   - None (void, no return)                   #
        #----------------------------------------------#
        
        ############
        #Initialize
        ##### Max epochs/iterations #####
        self.max_iters = max_iters

        ##### Warm-up parameters #####
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        ##### Additional inherit #####
        super().__init__(optimizer, last_epoch)


    ########################
    # Member Function List #
    ########################
    def get_lr(self) -> List[float]:
        #------------------------------------------------#
        # Description: Get/Query learning rate values    #
        # Input type:                                    #
        #   - None (void, no input)                      #  
        # Return type:                                   #
        #   - List[float] (updated learning rate values) #
        #------------------------------------------------#
        from math import cos, pi #Temporal import math cosine, pi

        ############
        #Initialize
        ##### Warm-up factor #####
        warmup_factor = self._get_warmup_factor_at_iter(method = self.warmup_method, \
                                                        iter = self.last_epoch, \
                                                        warmup_iters = self.warmup_iters, \
                                                        warmup_factor = self.warmup_factor)
        
        return [(base_lr * warmup_factor * 0.5 * (1.0 + cos(pi * self.last_epoch / \
                 self.max_iters))) for base_lr in self.base_lrs]


    def _compute_values(self) -> List[float]:
        #-----------------------------------------------#
        # Description: Compute learning rate values     #
        # Input type:                                   #
        #   - None (void, no input)                     #  
        # Return type:                                  #
        #   - List[float] (output learning rate values) #
        #-----------------------------------------------#

        return self.get_lr()
    

    def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, \
                                   warmup_factor: float) -> float:
        #--------------------------------------------------------------------#
        # Description: Get/Query warm-up factors at specific epoch/iteration #
        # Input type:                                                        #
        #   - str (self-defined methods)                                     #
        #   - int (epoch/iteration)                                          #
        #   - int (warmup_iters)                                             #
        #   - float (warmup_factor)                                          #
        # Return type:                                                       #
        #   - float (output learning rate factors)                           #
        #--------------------------------------------------------------------#

        #################################
        #Customized self-defined methods
        if (iter >= warmup_iters):
            return 1.0
        else:
            pass
        
        if (method == "constant"):
            return warmup_factor
        
        elif (method == "linear"):
            alpha = (iter / warmup_iters)

            return ((warmup_factor * (1 - alpha)) + alpha)
        else:
            raise ValueError("Unknown warmup method: {}".format(method))



#=================#
# Mapper Function #
#=================#
def Scheduler_Mapper(scheduler: str, optimizer: Optimizer, step: Any, \
                     ratio: float, epochs: int) -> _LRScheduler:
    #--------------------------------------------------------------#
    # Description: Customized mapper for self-defined lr scheduler #
    # Input type:                                                  #
    #   - str (self-defined lr scheduler)                          #
    #   - Optimizer (self-defined optimizer)                       #
    #   - Any (epoch step)                                         #
    #   - float (scheduler/decay ratio)                            #
    #   - int (total/whole epochs/iterations)                      #
    # Return type:                                                 #
    #   - _LRScheduler (result lr scheduler)                       #
    #--------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    #############################################
    #Mapper process with different lr schedulers
    ##### Normal-Step #####
    mapper_dict['step_lr'] = StepLR(optimizer = optimizer, step_size = step, \
                                    gamma = ratio)
    ##### Exponential #####
    mapper_dict['exponential_lr'] = ExponentialLR(optimizer = optimizer, gamma = ratio)

    ##### Multi-Step #####
    mapper_dict['multistep_lr'] = MultiStepLR(optimizer = optimizer, milestones = step, \
                                              gamma = ratio)
    ##### Warm-Up #####
    mapper_dict['warmupmultistep_lr'] = LRMultiplier(optimizer = optimizer, \
                                                      multiplier = WarmupParamScheduler(scheduler = \
                                                       MultiStepParamScheduler(values = [(0.1 ** k) for \
                                                       k in range(len([x for x in step if (x <= epochs)]) + 1)], \
                                                       milestones = [x for x in step if (x <= epochs)], \
                                                       num_updates = epochs), warmup_factor = (1.0 / 100), \
                                                       warmup_length = min((1000 / epochs), 1.0), \
                                                       warmup_method = "linear"), \
                                                      max_iter = epochs)
    
    mapper_dict['warmupcosine_lr'] = LRMultiplier(optimizer = optimizer, \
                                                   multiplier = WarmupParamScheduler(scheduler = \
                                                    CosineParamScheduler(start_value = 1, end_value = 0), \
                                                    warmup_factor = (1.0 / 100), warmup_length = \
                                                    min((1000 / epochs), 1.0), warmup_method = "linear"), \
                                                   max_iter = epochs)
    
    return mapper_dict[scheduler]