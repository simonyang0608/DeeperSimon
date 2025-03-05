#************************************************************#
# Source: Generator.py                                       #
#                                                            #
# Description: Generator for config/parameters, scripts, ... #
#                                                            #
# Author: SimonYang                                          #
#************************************************************#

#================#
# Import Section #
#================#
#############
#Config yaml
from yaml import safe_load, dump

######################
#Regularize (i.e. re)
from re import compile

########
#Shutil
from shutil import copy

############################
#Operating system (i.e. OS)
from os import listdir, mkdir
from os.path import join, exists


#===================#
# Global Initialize #
#===================#
#########################################################
#Source file path for config .yaml, scripts, models, ...
##### Config #####
src_cfg = "/opt/getac/AnomalyDetection_MemSeg/Config/LIOHO/Baking_Paint/Satellite_Pedestal/Segmentation/Defect.yaml"

##### Models #####
src_models = "/mnt/HDD/DataCenter/LIOHO/Models/SP-03/Segmentation/Global"

##### Scripts #####
src_scripts = "/mnt/HDD/DataCenter/LIOHO/Scripts/SP-03/Segmentation/Global"

src_train = "/opt/getac/AnomalyDetection_MemSeg/Template_Shell/train.sh"
src_evaluate = "/opt/getac/AnomalyDetection_MemSeg/Template_Shell/evaluate.sh"
src_export = "/opt/getac/AnomalyDetection_MemSeg/Template_Shell/export.sh"



#======================#
# Define Function List #
#======================#
def Generator_Worker(src_cfg: str, src_models: str, src_scripts: str, \
                     src_train: str, src_evaluate: str, src_export: str) -> None:
    #-----------------------------------------------------#
    # Description: Generator worker details pipeline/flow #
    # Input type:                                         #
    #   - str (source config file)                        #
    #   - str (source models)                             #
    #   - str (source scripts)                            #
    #   - str (source train shell)                        #
    #   - str (source evaluate shell)                     #
    #   - str (source export shell)                       #
    # Return type:                                        #
    #   - None (void, no return)                          #
    #-----------------------------------------------------#

    ############
    #Initialize
    ##### Re compile/organize #####
    general_compile = compile(r'\d+')

    ##### Version type #####
    type_ver = "V"

    ##### Query/Get latest scripts folder versions in project path #####
    list_scripts_ver = listdir(src_scripts)
    sort_scripts_ver = sorted(list_scripts_ver, key = lambda s: \
                              int((general_compile.search(s)).group()))
    latest_scripts_ver = (sort_scripts_ver[-1])[1:]
    next_scripts_ver = str(int(latest_scripts_ver) + 1)

    ##### Query/Get latest models folder versions in project path #####
    list_models_ver = listdir(src_models)
    sort_models_ver = sorted(list_models_ver, key = lambda s: \
                             int((general_compile.search(s)).group()))
    latest_models_ver = (sort_models_ver[-1])[1:]

    ##### Current models versions directory #####
    curr_models_ver_path = join(src_models, (type_ver + latest_models_ver))

    ##### Next scripts, models versions directory #####
    next_scripts_ver_path = join(src_scripts, (type_ver + next_scripts_ver))
    next_models_ver_path = join(src_models, (type_ver + next_scripts_ver))


    ####################
    #Whole process/flow
    ##### Step 1.1: Make directory for scripts #####
    if (not (exists(next_scripts_ver_path))):
        mkdir(next_scripts_ver_path)
    else:
        pass
    
    ##### Step 1.2: Make directory for models mapped from scripts #####
    if (not (exists(next_models_ver_path))):
        mkdir(next_models_ver_path)
    else:
        pass

    ##### Step 2: Modify output directory in config yaml file #####
    copy(src_cfg, join(next_scripts_ver_path, "Config.yaml")) #Copy from source to target

    with open(join(next_scripts_ver_path, "Config.yaml")) as fyaml_open: #Load
        list_info = safe_load(fyaml_open)

    list_info["OUTPUT"] = next_models_ver_path #Replace

    with open(join(next_scripts_ver_path, "Config.yaml"), "w") as fyaml_open: #Overwrite
        dump(list_info, fyaml_open)


    ##### Step 3.1: Modify model, config informations in train shell script #####
    copy(src_train, join(next_scripts_ver_path, "train.sh")) #Copy from source to target

    with open(join(next_scripts_ver_path, "train.sh"), 'r') as fsh_open: #Read
        data_info = fsh_open.read()

        sh_replace = data_info.replace("best_checkpoint.pth", join(curr_models_ver_path, \
                                       "train_result", "best_checkpoint.pth"))
        sh_replace = sh_replace.replace(src_cfg, join(next_scripts_ver_path, "Config.yaml"))
        
    with open(join(next_scripts_ver_path, "train.sh"), 'w') as fsh_open: #Overwrite
        fsh_open.write(sh_replace)


    ##### Step 3.2: Modify model, config informations in evaluate shell script #####
    copy(src_evaluate, join(next_scripts_ver_path, "evaluate.sh")) #Copy from source to target

    with open(join(next_scripts_ver_path, "evaluate.sh"), 'r') as fsh_open: #Read
        data_info = fsh_open.read()

        sh_replace = data_info.replace("best_checkpoint.pth", join(next_models_ver_path, \
                                       "train_result", "best_checkpoint.pth"))
        sh_replace = sh_replace.replace(src_cfg, join(next_scripts_ver_path, "Config.yaml"))
        
    with open(join(next_scripts_ver_path, "evaluate.sh"), 'w') as fsh_open: #Overwrite
        fsh_open.write(sh_replace)


    ##### Step 3.3: Modify model, config informations in export shell script #####
    copy(src_export, join(next_scripts_ver_path, "export.sh")) #Copy from source to target

    with open(join(next_scripts_ver_path, "export.sh"), 'r') as fsh_open: #Read
        data_info = fsh_open.read()

        sh_replace = data_info.replace("best_checkpoint.pth", join(next_models_ver_path, \
                                       "train_result", "best_checkpoint.pth"))
        sh_replace = sh_replace.replace(src_cfg, join(next_scripts_ver_path, "Config.yaml"))
        
    with open(join(next_scripts_ver_path, "export.sh"), 'w') as fsh_open: #Overwrite
        fsh_open.write(sh_replace)



def Generator(src_cfg: str, src_models: str, src_scripts: str, \
              src_train: str, src_evaluate: str, src_export: str) -> None:
    #-----------------------------------------------------#
    # Description: Generator function for training script #
    # Input type:                                         #
    #   - str (source config file)                        #
    #   - str (source models)                             #
    #   - str (source scripts)                            #
    #   - str (source train shell)                        #
    #   - str (source evaluate shell)                     #
    #   - str (source export shell)                       #
    # Return type:                                        #
    #   - None (void, no return)                          #
    #-----------------------------------------------------#

    ########################
    #Generator process/flow
    Generator_Worker(src_cfg = src_cfg, src_models = src_models, src_scripts = src_scripts, \
                     src_train = src_train, src_evaluate = src_evaluate, src_export = src_export)



#========================#
# Setup Code Entry Point #
#========================#
if (__name__ == "__main__"):
    Generator(src_cfg = src_cfg, src_models = src_models, src_scripts = src_scripts, \
              src_train = src_train, src_evaluate = src_evaluate, src_export = src_export) #Generator function call
