#!/bin/bash

##################
#Initial settings
##### Main script/source code #####
MAIN_SRC="/opt/getac/AnomalyDetection_MemSeg/Main.py"

##### Config file location #####
CONFIG_FILE="/opt/getac/AnomalyDetection_MemSeg/Config/DQA/Suspenders/Detection/Defect.yaml"

##### Defect file location #####
DEFECT_FILE='/opt/getac/Defect_Definitions.txt'

##### Model checkpoint/weight location #####
MODEL_CKPT="/mnt/HDD/DataCenter/LIOHO/Models/tmp/train_result/checkpoint.pth"

##### Learning rate #####
LR=0.0001

##### Epochs/Iterations #####
EPOCHS=7255


####################################################
#Self-defined/customized different workflow/process
##### Train model from original random weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --lr $LR"

##### Train model from pretrained weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --pretrained $MODEL_CKPT --lr $LR"

##### Train model from resume weight/parameters #####
MAIN_FLOW="python3 $MAIN_SRC -train --epochs $EPOCHS --config $CONFIG_FILE --defect $DEFECT_FILE --resume $MODEL_CKPT --lr $LR"

##### Evaluate model from trained weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -eval --config $CONFIG_FILE --defect $DEFECT_FILE --eval_ckpt $MODEL_CKPT"

##### Export model from trained weight/parameters #####
#MAIN_FLOW="python3 $MAIN_SRC -export --config $CONFIG_FILE --defect $DEFECT_FILE --export_ckpt $MODEL_CKPT"



##############################
#Main process/flow executions
$MAIN_FLOW