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
##### Train model from original random weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --lr $LR"

##### Train model from pretrained weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --pretrained $MODEL_CKPT --lr $LR"

##### Train model from resume weight/parameters #####
MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --resume $MODEL_CKPT --lr $LR"



##############################
#Main process/flow executions
$MAIN_FLOW
