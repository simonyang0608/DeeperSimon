#!/bin/bash

##################
#Initial settings
##### Main script/source code #####
MAIN_SRC="/opt/getac/AnomalyDetection_MemSeg/Main.py"

##### Config file location #####
CONFIG_FILE="/opt/getac/AnomalyDetection_MemSeg/Config/LIOHO/Baking_Paint/Satellite_Pedestal/Segmentation/Defect.yaml"

##### Defect file location #####
DEFECT_FILE='/opt/getac/Defect_Definitions.txt'

##### Model checkpoint/weight location #####
MODEL_CKPT="best_checkpoint.pth"

##### Learning rate #####
LR=0.0001

##### Epochs/Iterations #####
EPOCHS=150


####################################################
#Self-defined/customized different workflow/process
##### Evaluate model from trained weight/parameters #####
MAIN_FLOW="python3 $MAIN_SRC -eval --config $CONFIG_FILE --defect $DEFECT_FILE --eval_ckpt $MODEL_CKPT"



##############################
#Main process/flow executions
$MAIN_FLOW
