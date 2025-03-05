#!/bin/bash

##################
#Initial settings
##### Main script/source code #####
MAIN_SRC="/opt/getac/AnomalyDetection_MemSeg/Generator.py"


###############################################
#LIOHO training pipeline/workflow/main process
MAIN_FLOW="python3 $MAIN_SRC"



##############################
#Main process/flow executions
$MAIN_FLOW