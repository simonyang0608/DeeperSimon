#**********************************************************************#
# Source: Loss_Function.py                                             #
#                                                                      #
# Description: Customized loss functions for model training/validation #
#                                                                      #
# Author: SimonYang                                                    #
#**********************************************************************#

#================#
# Import Section #
#================#
##################
#Pytorch exp, ...
from torch import exp, nonzero

################################################
#Pytorch nn module (i.e. basic inherit), tensor
from torch.nn import Module
from torch import Tensor

############################################################
#Pytorch nn loss functions (i.e. bce-loss, focal-loss, ...)
from torch.nn import (BCEWithLogitsLoss, CrossEntropyLoss, \
                      BCELoss)

###############################
#Pytorch functionals functions
from torch.nn.functional import (binary_cross_entropy, softmax, \
                                 cross_entropy)


#=====================#
# Class Function List #
#=====================#
class DiceLoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - None (void, no input)                 #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(DiceLoss, self).__init__() #Inherit from torch.nn.module basis


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor, \
                smooth: int = 1) -> float:
        #----------------------------------------------#
        # Description: Feed-forward pass for dice loss #
        # Input type:                                  #
        #   - Tensor (input featuremap)                #
        #   - Tensor (target featuremap)               #
        #   - int (smooth co-efficient/value)          #
        # Return type:                                 #
        #   - float (result loss value)                #
        #----------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Calculate measured meter #####
        intersection = ((target_feat * inpt_feat).sum([1, 2, 3]))
        union = (target_feat.sum([1, 2, 3]) + inpt_feat.sum([1, 2, 3]))

        ##### Step 2: Calculate loss value #####
        dice = (((2. * intersection) + smooth) / (union + smooth)).mean()

        return (1 - dice)



class Dice_BCELoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - None (void, no input)                 #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Dice_BCELoss, self).__init__() #Inherit from torch.nn.module basis

        ##### Dice loss #####
        self.dice_loss = DiceLoss()


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor) -> float:
        #--------------------------------------------------#
        # Description: Feed-forward pass for dice+bce loss #
        # Input type:                                      #
        #   - Tensor (input featuremap)                    #
        #   - Tensor (target featuremap)                   #
        # Return type:                                     #
        #   - float (result loss value)                    #
        #--------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Pre-process #####
        bce_loss = (binary_cross_entropy(input = inpt_feat, target = target_feat, \
                                         reduction = 'sum') / len(nonzero(target_feat)))

        ##### Step 2: Calculate loss value #####
        dice_loss = self.dice_loss(inpt_feat, target_feat)

        return (bce_loss + dice_loss)
    


class Dice_FocalLoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, alpha: float, gamma: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - float (alpha value)                   #
        #   - int (gamma value)                     #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Dice_FocalLoss, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Alpha/Gamma value #####
        self.alpha = alpha
        self.gamma = gamma

        ##### Dice loss #####
        self.dice_loss = DiceLoss()


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor) -> float:
        #---------------------------------------------------#
        # Description: Feed-forward pass for focal+bce loss #
        # Input type:                                       #
        #   - Tensor (input featuremap)                     #
        #   - Tensor (target featuremap)                    #
        # Return type:                                      #
        #   - float (result loss value)                     #
        #---------------------------------------------------#
        alpha, gamma = self.alpha, self.gamma #Alpha/Gamma value initialize

        ####################
        #Whole process/flow
        ##### Step 1: Pre-measure/calculate loss #####
        bce_loss = binary_cross_entropy(input = inpt_feat, target = target_feat, \
                                        reduction="none")
        
        tmp_feat = ((inpt_feat * target_feat) + ((1 - inpt_feat) * (1 - target_feat)))
        focal_loss = (bce_loss * ((1 - tmp_feat) ** gamma))

        ##### Step 2: Calculate loss value #####
        tmp_alpha = ((alpha * target_feat) + ((1 - alpha) * (1 - target_feat)))
        focal_loss = (((tmp_alpha * focal_loss).sum()) / len(nonzero(target_feat)))

        dice_loss = self.dice_loss(inpt_feat, target_feat)

        return (focal_loss + dice_loss)



class IOULoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - None (void, no input)                 #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(IOULoss, self).__init__() #Inherit from torch.nn.module basis


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor, \
                smooth: int = 1) -> float:
        #---------------------------------------------#
        # Description: Feed-forward pass for iou loss #
        # Input type:                                 #
        #   - Tensor (input featuremap)               #
        #   - Tensor (target featuremap)              #
        #   - int (smooth co-efficient/value)         #
        # Return type:                                #
        #   - float (result loss value)               #
        #---------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Flatten #####
        tmp_feat = inpt_feat.view(-1)
        target_feat = target_feat.view(-1)

        ##### Step 2: Calculate loss value #####
        intersection = ((tmp_feat * target_feat).sum())

        total = ((tmp_feat + target_feat).sum())

        union = (total - intersection)

        iou = ((intersection + smooth) / (union + smooth))

        return (1 - iou)
    


class Focal_BCELoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, alpha: float, gamma: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - float (alpha value)                   #
        #   - int (gamma value)                     #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Focal_BCELoss, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Alpha/Gamma value #####
        self.alpha = alpha
        self.gamma = gamma


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor) -> float:
        #---------------------------------------------------#
        # Description: Feed-forward pass for focal+bce loss #
        # Input type:                                       #
        #   - Tensor (input featuremap)                     #
        #   - Tensor (target featuremap)                    #
        # Return type:                                      #
        #   - float (result loss value)                     #
        #---------------------------------------------------#
        alpha, gamma = self.alpha, self.gamma #Alpha/Gamma value initialize

        ####################
        #Whole process/flow
        ##### Step 1: Flatten #####
        tmp_feat = inpt_feat.view(-1)
        target_feat = target_feat.view(-1)

        ##### Step 2: Calculate loss value #####
        bce = binary_cross_entropy(input = tmp_feat, target = target_feat, \
                                   reduction = 'mean')

        bce_exp = exp(((-1) * bce))

        return (alpha * ((1 - bce_exp) ** gamma) * bce)
    


class Focal_CELoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, alpha: float, gamma: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - float (alpha value)                   #
        #   - int (gamma value)                     #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Focal_CELoss, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Alpha/Gamma value #####
        self.alpha = alpha
        self.gamma = gamma


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor) -> float:
        #--------------------------------------------------#
        # Description: Feed-forward pass for focal+ce loss #
        # Input type:                                      #
        #   - Tensor (input featuremap)                    #
        #   - Tensor (target featuremap)                   #
        # Return type:                                     #
        #   - float (result loss value)                    #
        #--------------------------------------------------#
        alpha, gamma = self.alpha, self.gamma #Alpha/Gamma value initialize

        ####################
        #Whole process/flow
        ##### Step 1: Pre-process #####
        tmp_feat = softmax(inpt_feat)

        ##### Step 2: Flatten #####
        tmp_feat = tmp_feat.view(-1)
        target_feat = target_feat.view(-1)

        ##### Step 3: Calculate loss value #####
        ce = cross_entropy(input = tmp_feat, target = target_feat, \
                           reduction = 'mean')

        ce_exp = exp(((-1) * ce))

        return (alpha * ((1 - ce_exp) ** gamma) * ce)
    


class TverskyLoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, alpha: float, beta: float) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - float (alpha value)                   #
        #   - float (beta value)                    #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(TverskyLoss, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Alpha/Beta value #####
        self.alpha = alpha
        self.beta = beta


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor, \
                smooth: int = 1) -> float:
        #-------------------------------------------------#
        # Description: Feed-forward pass for tversky loss #
        # Input type:                                     #
        #   - Tensor (input featuremap)                   #
        #   - Tensor (target featuremap)                  #
        #   - int (smooth co-efficient/value)             #
        # Return type:                                    #
        #   - float (result loss value)                   #
        #-------------------------------------------------#
        alpha, beta = self.alpha, self.beta #Alpha/Beta value initialize

        ####################
        #Whole process/flow
        ##### Step 1: Flatten #####
        tmp_feat = inpt_feat.view(-1)
        target_feat = target_feat.view(-1)

        ##### Step 2: Calculate loss value #####
        tp = ((tmp_feat * target_feat).sum())
        fp = ((tmp_feat * (1 - target_feat)).sum())

        fn = (((1 - tmp_feat) * target_feat).sum())

        tversky = ((tp + smooth) / (tp + (alpha * fp) + (beta * fn) + \
                    smooth))
        
        return (1 - tversky)
    


class Focal_TverskyLoss(Module):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, alpha: float, beta: float, gamma: int) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - float (alpha value)                   #
        #   - float (beta value)                    #
        #   - int (gamma value)                     #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#
        super(Focal_TverskyLoss, self).__init__() #Inherit from torch.nn.module basis

        ############
        #Initialize
        ##### Alpha/Beta value #####
        self.alpha = alpha
        self.beta = beta

        ##### Gamma value #####
        self.gamma = gamma


    ####################
    # Forward Function #
    ####################
    def forward(self, inpt_feat: Tensor, target_feat: Tensor, \
                smooth: int = 1) -> float:
        #-------------------------------------------------------#
        # Description: Feed-forward pass for focal+tversky loss #
        # Input type:                                           #
        #   - Tensor (input featuremap)                         #
        #   - Tensor (target featuremap)                        #
        #   - int (smooth co-efficient/value)                   #
        # Return type:                                          #
        #   - float (result loss value)                         #
        #-------------------------------------------------------#
        alpha, beta = self.alpha, self.beta #Alpha/Beta value initialize
        gamma = self.gamma #Gamma value initialize

        ####################
        #Whole process/flow
        ##### Step 1: Flatten #####
        tmp_feat = inpt_feat.view(-1)
        target_feat = target_feat.view(-1)

        ##### Step 3: Calculate loss value #####
        tp = ((tmp_feat * target_feat).sum())
        fp = ((tmp_feat * (1 - target_feat)).sum())

        fn = (((1 - tmp_feat) * target_feat).sum())

        tversky = ((tp + smooth) / (tp + (alpha * fp) + (beta * fn) + \
                    smooth))
        
        return ((1 - tversky) ** gamma)
    


#=================#
# Mapper Function #
#=================#
def Loss_Mapper(loss: str, alpha: float, beta: float, \
                gamma: int) -> Module:
    #---------------------------------------------------------------#
    # Description: Customized mapper for self-defined loss function #
    # Input type:                                                   #
    #   - str (self-defined loss function)                          #
    #   - float (alpha value)                                       #
    #   - float (beta value)                                        #
    #   - int (gamma value)                                         #
    # Return type:                                                  #
    #   - Module (result loss function)                             #
    #---------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ##############################################
    #Mapper process with different loss functions
    ##### BCE #####
    mapper_dict['bce_logits'] = BCEWithLogitsLoss()
    mapper_dict['focal_bce'] = Focal_BCELoss(alpha = alpha, \
                                             gamma = gamma)
    mapper_dict['bce'] = BCELoss()

    ##### CE #####
    mapper_dict['ce'] = CrossEntropyLoss()
    mapper_dict['focal_ce'] = Focal_CELoss(alpha = alpha, \
                                           gamma = gamma)
    ##### DICE #####
    mapper_dict['dice'] = DiceLoss()
    mapper_dict['dice_bce'] = Dice_BCELoss()
    mapper_dict['dice_focal'] = Dice_FocalLoss(alpha = alpha, \
                                               gamma = gamma)
    ##### IOU #####
    mapper_dict['iou'] = IOULoss()

    ##### Tversky #####
    mapper_dict['tversky'] = TverskyLoss(alpha = alpha, \
                                         beta = beta)
    mapper_dict['focal_tversky'] = Focal_TverskyLoss(alpha = alpha, \
                                                     beta = beta, \
                                                     gamma = gamma)
    
    return mapper_dict[loss]