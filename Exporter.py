#**************************************************************#
# Source: Exporter.py                                          #
#                                                              #
# Description: Major/Main exporter for model export/conversion #
#                                                              #
# Author: SimonYang                                            #
#**************************************************************#

#================#
# Import Section #
#================#
########################################
#Pytorch nn module (i.e. basic inherit)
from Model.Global_Builder.Module import Module

#####################
#Pytorch onnx export
from torch.onnx import export

#########################
#Pytorch randoms, device
from torch import randn, device

############################
#Operating system (i.e. OS)
from os.path import join

####################
#Typing format list
from Model.Global_Builder.Module import Any


#======================#
# Define Function List #
#======================#
def ONNX_Exporter(logger: Any, fpath: str, model: Module, channel: int, \
                  height: int, width: int, device: device, input_names: str, \
                  output_names: str, version: int, task: str, \
                  batch_size: int = 1) -> None:
    #------------------------------------------------------------------------#
    # Description: Convert pytorch .pth model file to .onnx format to export #
    # Input type:                                                            #
    #   - Any (logging record)                                               #
    #   - str (file path)                                                    #
    #   - Module (self-defined model)                                        #
    #   - int (input channel)                                                #
    #   - int (input height)                                                 #
    #   - int (input width)                                                  #
    #   - device (gpu/cpu device)                                            #
    #   - str (input names)                                                  #
    #   - str (output names)                                                 #
    #   - int (opset version)                                                #
    #   - str (input tasks type)                                             #
    #   - int (input batch-size)                                             #
    # Return type:                                                           #
    #   - None (void, no return)                                             #
    #------------------------------------------------------------------------#
    model.eval() #Switch model to validate mode

    ####################
    #Whole process/flow
    ##### Step 1: Create random/dummy input tensor #####
    inpt_tensor = randn(batch_size, channel, height, width).to(device)
    
    if (task == 'detection'):
        inpt_dummy = (inpt_tensor, None)
    else:
        inpt_dummy = (inpt_tensor)
    
    
    ##### Step 2: Export to onnx format model file #####
    logger.info('==> Using input names: {}'.format(input_names))
    logger.info('==> Using output names: {}'.format(output_names))

    logger.info('==> Using version: {}'.format(version))

    export(model = model, args = inpt_dummy, f = join(fpath, 'model_torch.onnx'), \
           verbose = True, input_names = input_names, output_names = output_names, \
           opset_version = version, keep_initializers_as_inputs = True)
    
    logger.info('===> Done! Convert to onnx model format completed!!')



#=================#
# Mapper Function #
#=================#
def Exporter_Mapper(cfg: dict, logger: Any, fpath: str, model: Module, \
                    device: device) -> Any:
    #---------------------------------------------------------------------#
    # Description: Customized mapper for self-defined export model format #
    # Input type:                                                         #
    #   - dict (whole config information)                                 #
    #   - Any (logging record)                                            #
    #   - str (file path)                                                 #
    #   - Module (self-defined model)                                     #
    #   - device (gpu/cpu device)                                         #
    # Return type:                                                        #
    #   - Any (exporter function call)                                    #
    #---------------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    #########################################
    #Mapper process with different exporters
    ##### ONNX #####
    mapper_dict['onnx'] = ONNX_Exporter(logger = logger, fpath = fpath, model = model, \
                                        channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                                        height = cfg['INPUT']['RESOLUTION']['HEIGHT'], \
                                        width = cfg['INPUT']['RESOLUTION']['WIDTH'], device = \
                                        device, input_names = cfg['EXPORT']['ONNX']['INPUT_NAMES'], \
                                        output_names = cfg['EXPORT']['ONNX']['OUTPUT_NAMES'], \
                                        version = cfg['EXPORT']['ONNX']['OPSET_VERSION'], \
                                        task = cfg['MODEL']['TASK'])
    
    return mapper_dict[cfg['EXPORT']['FORMAT']]