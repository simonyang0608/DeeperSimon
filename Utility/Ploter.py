#***************************************************************************#
# Source: Ploter.py                                                         #
#                                                                           #
# Description: Visualize/Plot output results from model training/validation #
#                                                                           #
# Author: SimonYang                                                         #
#***************************************************************************#

#================#
# Import Section #
#================#
#######################
#Tensorboard visualize
from tensorboardX import SummaryWriter

#########################
#Torchvision figure grid
from torchvision.utils import make_grid

################################################
#Pytorch nn module (i.e. basic inherit), tensor
from torch.nn import Module
from torch import Tensor

#############
#Numpy array
from numpy import array

###############################
#Matplotlib, mlxtend functions
from matplotlib.pyplot import (plot, xlabel, ylabel, \
                               title, legend, figure)
from mlxtend.plotting import plot_confusion_matrix


#======================#
# Define Function List #
#======================#
def Plot_AUROC_Curve(fp_rate: float, tp_rate: float, auroc_score: float) -> figure:
    #---------------------------------#
    # Description: Plot AUROC curve   #
    # Input type:                     #
    #   - float (false positive rate) #
    #   - float (true positive rate)  #
    #   - float (auroc score value)   #
    # Return type:                    #
    #   - figure (result figure)      #
    #---------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Construct figure, title #####
    rest_figure = figure(1)
    title('AUROC Curve')

    ##### Step 2: Setup coordinates #####
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')

    ##### Step 3: Plot measured meter/parameters #####
    plot(fp_rate, tp_rate, marker = '.', \
         label = 'AUROC(Area = {:.3f})'.format(auroc_score))

    legend(loc = 'best')

    return rest_figure



def Plot_Confusion_Matrix(confusion_matrix: array) -> figure:
    #------------------------------------#
    # Description: Plot confusion matrix #
    # Input type:                        #
    #   - array (confusion matrix)       #
    # Return type:                       #
    #   - figure (result figure)         #
    #------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Construct figure, title #####
    rest_figure, _ = plot_confusion_matrix(conf_mat = confusion_matrix)
    
    title('Confusion Matrix')

    ##### Step 2: Setup coordinates #####
    xlabel('Predictions')
    ylabel('Actuals')

    return rest_figure



def Add_Scalar(writer: SummaryWriter, title: str, value: float, \
               epoch: int) -> None:
    #---------------------------------------------------#
    # Description: Single learning curve visualization  #
    # Input type:                                       #
    #   - SummaryWriter (visualized data writer)        #
    #   - str (figure title)                            #
    #   - float (measured value)                        #
    #   - int (epoch/iteration)                         #
    # Return type:                                      #
    #   - None (void, no return)                        #
    #---------------------------------------------------#

    #######################
    #Learning curve scalar
    writer.add_scalar(tag = title, scalar_value = value, \
                      global_step = epoch)
    


def Add_Scalars(writer: SummaryWriter, title: str, dict_value: dict, \
                epoch: int) -> None:
    #-----------------------------------------------------#
    # Description: Multiple learning curve visualization  # 
    # Input type:                                         #
    #   - SummaryWriter (visualized data writer)          #
    #   - str (figure title)                              #
    #   - dict (measured value in dictionary type/format) #
    #   - int (epoch/iteration)                           #
    # Return type:                                        #
    #   - None (void, no return)                          #
    #-----------------------------------------------------#

    ########################
    #Learning curve scalars
    writer.add_scalars(main_tag = title, tag_scalar_dict = dict_value, \
                       global_step = epoch)
    


def Add_Image(writer: SummaryWriter, title: str, image: Tensor, \
              epoch: int) -> None:
    #--------------------------------------------#
    # Description: Image/Figure visualization    #
    # Input type:                                #
    #   - SummaryWriter (visualized data writer) #
    #   - str (figure title)                     #
    #   - Tensor (image/figure tensor)           #
    #   - int (epoch/iteration)                  #
    # Return type:                               #
    #   - None (void, no return)                 #
    #--------------------------------------------#
    
    ###################
    #Image/Figure grid
    image_grid = make_grid(tensor = image, nrow = 1, padding = 11, \
                           pad_value = 1.)
    
    writer.add_image(tag = title, img_tensor = image_grid, \
                     global_step = epoch)
    


def Add_Graph(writer: SummaryWriter, model: Module, image: Tensor) -> None:
    #--------------------------------------------#
    # Description: Model graph visualization     #
    # Input type:                                #
    #   - SummaryWriter (visualized data writer) #
    #   - Module (model architecture)            #
    #   - Tensor (image/figure tensor)           #
    # Return type:                               #
    #   - None (void, no return)                 #
    #--------------------------------------------#

    ##########################
    #Model architecture graph
    writer.add_graph(model = model, input_to_model = image)



def Add_Figure(writer: SummaryWriter, title: str, figure: figure, \
               epoch: int) -> None:
    #--------------------------------------------#
    # Description: Image/Figure visualization    #
    # Input type:                                #
    #   - SummaryWriter (visualized data writer) #
    #   - str (figure title)                     #
    #   - figure (image/figure)                  #
    #   - int (epoch/iteration)                  #
    # Return type:                               #
    #   - None (void, no return)                 #
    #--------------------------------------------#

    ##############
    #Image/Figure
    writer.add_figure(tag = title, figure = figure, global_step = epoch)