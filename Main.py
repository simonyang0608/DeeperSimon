#************************************************************************#
# Source: Main.py                                                        #
#                                                                        #
# Description: Main pipeline/flow for train/valid, evaluate, export, ... #
#                                                                        #
# Author: SimonYang                                                      #
#************************************************************************#

#================#
# Import Section #
#================#
######################
#Pytorch device, cuda
from torch import device
from torch.cuda import is_available

#############################
#Pyorch model/model measurer
from Model.Meta_Networker.General_Mapper import Model_Mapper
from Utility.Measurer import Summarizer

###################
#Logging, Colorlog
from logging import (getLogger, FileHandler, StreamHandler, \
                     Formatter, INFO)
from colorlog import ColoredFormatter

#############
#Config yaml
from yaml import load, dump, FullLoader

#################
#Argument parser
from argparse import ArgumentParser

############################
#Operating system (i.e. OS)
from os import mkdir
from os.path import join, exists

###################
#System (i.e. sys)
from sys import stdout

####################
#Typing format list
from typing import Any


#===================#
# Global Initialize #
#===================#
#################
#Argument parser
parser = ArgumentParser(description = 'Anomaly/Defect detection AI engine') #Describe initialize

##### Learning rate #####
parser.add_argument('--lr', default = 0.0001, type = float, metavar = 'LR', \
                    help = 'learning rate')

##### Train/Evaluate/Export mode #####
parser.add_argument('-train', action = 'store_true', help = 'train model on train/val set')
parser.add_argument('-eval', action = 'store_true', help = 'evaluate model on evaluate set')
parser.add_argument('-export', action = 'store_true', help = 'export model to desired format')

##### Pretrained/Resume mode #####
parser.add_argument('--pretrained', default = '', type = str, metavar = 'PATH', \
                    help = 'path to pretrained model checkpoint')
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', \
                    help = 'path to latest model checkpoint')

##### Start epoch #####
parser.add_argument('--start_epoch', default = 0, type = int, metavar = 'N', \
                    help = 'start epoch point')

##### Total epochs #####
parser.add_argument('--epochs', default = 90, type = int, metavar = 'N', \
                    help = 'number of epochs')

##### Evaluate/Export checkpoint path #####
parser.add_argument('--eval_ckpt', default = '', type = str, metavar = 'PATH', \
                    help = 'path to model checkpoint for evaluate')
parser.add_argument('--export_ckpt', default = '', type = str, metavar = 'PATH', \
                    help = 'path to model checkpoint for export')

##### Config yaml file path #####
parser.add_argument('--config', default = '', type = str, metavar = 'PATH', \
                    help = 'path to config yaml file')

##### Defect text file path #####
parser.add_argument('--defect', default = '', type = str, metavar = 'PATH', \
                    help = 'path to defect text file')

args = parser.parse_args() #Combine all arguments and construct


#####################
#Logging information
logger = getLogger('') #Initialize

logger.setLevel(INFO) #Set to INFO level



#======================#
# Define Function List #
#======================#
def Main_Worker(args: Any, logger: Any) -> None:
    #------------------------------------------------#
    # Description: Main worker details pipeline/flow #
    # Input type:                                    #
    #   - Any (arguments)                            #
    #   - Any (logging record)                       #
    # Return type:                                   #
    #   - None (void, no return)                     #
    #------------------------------------------------#

    ############
    #Initialize
    ##### Load yaml config file, and combined with arguments #####
    if (not args.config):
        print("Config yaml path can't be null !")

        return
    
    with open(args.config, 'r') as f_cfg:
        cfg = load(f_cfg, Loader = FullLoader)

    cfg.update(vars(args))

    ##### Logging information format #####
    formatter = Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', \
                            datefmt = '%a, %d %b %Y %H:%M:%S')

    sh = StreamHandler(stdout)
    sh.setFormatter(ColoredFormatter('%(log_color)s [%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', \
                                      datefmt = '%a, %d %b %Y %H:%M:%S'))
    
    ##### Defect definitions (e.g. self-defined file path/directory) #####
    defect_fpath = cfg['defect']
    defect_class_idx = 0
    defect_dict = {}

    with open(defect_fpath, 'r') as fopen:
        inter_data = (fopen.read()).split()

        for tmp_idx in range(0, len(inter_data), 2):
            if (int(inter_data[(tmp_idx + 1)])):
                defect_dict[(inter_data[tmp_idx])[:-1]] = defect_class_idx
                
                defect_class_idx += 1
                
            else:
                pass

    defect_classes = len(defect_dict)

    
    #################################################
    #Train/Validate, Evaluate, Export mode to switch
    ##### Train/Validate mode #####
    if (cfg['train']):
        target_fdir = join(cfg['OUTPUT'], 'train_result') #Target path for train/validate

        if (not (exists(target_fdir))): #Create directory if path not exists
            mkdir(target_fdir)
        else:
            pass

        #########################
        #Temporal import section
        ##### Random seed #####
        from random import seed

        ##### Pytorch seed, backend #####
        from torch import manual_seed
        from torch.backends import cudnn

        ##### Dataset loader #####
        from Data.Dataset_Mapper import Dataloader_Mapper

        ##### Loss function/criterion #####
        from Loss.Loss_Function import Loss_Mapper

        ##### Optimizer, lr scheduler #####
        from Optimizer.Optimizer import Optimizer_Mapper
        from Optimizer.LR_Scheduler import Scheduler_Mapper

        ##### Loop trainer #####
        from Trainer import Major_Loop_Trainer


        ####################
        #Whole process/flow
        ##### Step 1: Dump/Save whole config/args information #####
        with open(join(target_fdir, 'config.yaml'), 'w') as f_dump: #Config yaml file
            dump(cfg, f_dump)

        ##### Step 2: Logging information file path/stream #####
        fh = FileHandler(join(target_fdir, 'log_info.log'), mode = 'w+') #Log info file
        fh.setFormatter(formatter)

        logger.addHandler(fh) #Add file handler to logger
        logger.addHandler(sh) #Add stream handler to logger

        logger.info('*************************************** Train/Validate mode ***************************************') #Logger information

        for key in defect_dict:
            logger.info("|| {} : {}".format(key, (defect_dict[key] + 1))) #Logger information (i.e. defect class type)
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 3: Random seed distribution #####
        if (cfg['DEVICE']['SEED']):
            seed(cfg['DEVICE']['SEED']) #Random seed

            manual_seed(cfg['DEVICE']['SEED']) #Manual seed

            cudnn.deterministic = True
            cudnn.benchmark = False

            logger.warning('You have chosen to seed training. '
                           'This will turn on the CUDNN deterministic setting, '
                           'which can slow down your training considerably! '
                           'You may see unexpected behavior when restarting '
                           'from checkpoints.')
        else:
            pass
        
        ##### Step 4: GPU/CPU device #####
        if (is_available()): #Cuda available
            main_device = device('cuda:{}'.format(cfg['DEVICE']['GPU'])) #Point to GPU device

            logger.info('=> Using GPU device with cuda:{}'.format(cfg['DEVICE']['GPU']))

        else:
            main_device = device('cpu') #Point to CPU device

            logger.info('=> Using CPU device')

        logger.info("*-------------------------------------------------------------------*")

        ##### Step 5-1: Construct/Load model network/architecture #####
        if ((not cfg['MODEL']['BACKBONE']) or (not cfg['MODEL']['META_NETWORK'])): #Issue addressment
            logger.error("Model informations can't be null !")

            return
        
        logger.info('=> Start constructing {} model: {} ...'.format(cfg['MODEL']['TASK'], cfg['MODEL']['META_NETWORK']))
        logger.info('==> Using backbone: {}'.format(cfg['MODEL']['BACKBONE']))

        main_model = Model_Mapper(cfg = cfg, num_classes = defect_classes) #Construct model architecture

        main_model = main_model.to(main_device) #Point model to main device

        Summarizer(logging = logger, model = main_model, device = main_device, \
                   channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                   height = cfg['INPUT']['RESOLUTION']['HEIGHT'], \
                   width = cfg['INPUT']['RESOLUTION']['WIDTH'], \
                   task = cfg['MODEL']['TASK']) #Summary for model
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 5-2: Load customized model weight/checkpoint #####
        if (cfg['pretrained']): #Load pretrained model weight/checkpoint
            logger.info("=> Using pretrained model checkpoint: {}".format(cfg['pretrained']))

            from torch import load as torch_load #Temporal import pytorch load weight/checkpoint

            ckpt = torch_load(cfg['pretrained'], map_location = main_device) #Model checkpoint
            main_model.load_state_dict(ckpt['state_dict']) #Load checkpoint weight

            logger.info("*-------------------------------------------------------------------*")

        elif (cfg['resume']): #Load resume model weight/checkpoint
             logger.info("=> Using resume model checkpoint: {}".format(cfg['resume']))

             from torch import load as torch_load #Temporal import pytorch load weight/checkpoint

             ckpt = torch_load(cfg['resume'], map_location = main_device) #Model checkpoint

             cfg['start_epoch'] = ckpt['epoch'] #Load checkpoint epoch
             main_model.load_state_dict(ckpt['state_dict']) #Load checkpoint weight

             cfg['epochs'] = (cfg['start_epoch'] + cfg['epochs']) #Customized for whole epochs accumulate

             logger.info("==> Start at epoch: {}".format(ckpt['epoch']))
             logger.info("*-------------------------------------------------------------------*")

        else:
            pass

        ##### Step 5-3: Construct/Load loss, optimizer, scheduler functions #####
        if ((not cfg['LOSS']['FUNCTION']) or (not cfg['OPTIMIZER']['FUNCTION']) \
            or (not cfg['LR_SCHEDULER']['FUNCTION'])): #Issue addressment
            logger.error("Optimizing functions can't be null !")

            return

        logger.info('=> Using loss function: {}'.format(cfg['LOSS']['FUNCTION']))

        main_criterien = Loss_Mapper(loss = cfg['LOSS']['FUNCTION'], \
                                     alpha = cfg['LOSS']['ALPHA'], \
                                     beta = cfg['LOSS']['BETA'], \
                                     gamma = cfg['LOSS']['GAMMA']) #Construct loss function
        
        main_criterien = main_criterien.to(main_device) #Point criterien function to main device
        
        logger.info('=> Using optimizer: {}'.format(cfg['OPTIMIZER']['FUNCTION']))

        main_optimizer = Optimizer_Mapper(optimizer = cfg['OPTIMIZER']['FUNCTION'], \
                                          model = main_model, lr = cfg['lr'], \
                                          momentum = cfg['OPTIMIZER']['MOMENTUM'], \
                                          weight_decay = cfg['OPTIMIZER']['WEIGHT_DECAY']) #Construct optimizer function
        
        if (cfg['resume']): #Load resume optimizer
            main_optimizer.load_state_dict(ckpt['optimizer']) #Load checkpoint optimizer
        else:
            pass
        
        logger.info('=> Using lr scheduler: {}'.format(cfg['LR_SCHEDULER']['FUNCTION']))

        main_scheduler = Scheduler_Mapper(scheduler = cfg['LR_SCHEDULER']['FUNCTION'], \
                                          optimizer = main_optimizer, step = cfg['LR_SCHEDULER']['STEP'], \
                                          ratio = cfg['LR_SCHEDULER']['RATIO'], \
                                          epochs = cfg['epochs']) #Construct scheduler function
        
        if (cfg['resume']): #Load resume scheduler
            main_scheduler.load_state_dict(ckpt['scheduler']) #Load checkpoint scheduler
        else:
            pass
        
        logger.info("==> Base lr: {}".format(cfg['lr']))
        logger.info("*-------------------------------------------------------------------*")
        
        ##### Step 6: Create dataset loader #####
        main_mapper = Dataloader_Mapper(cfg = cfg, defect_dict = defect_dict) #Class object construct

        main_loader = main_mapper.train_val_loader(logging = logger, task = cfg['MODEL']['TASK'], \
                                                   batch_size = cfg['INPUT']['BATCH_SIZE'], \
                                                   format = cfg['DATASET']['FORMAT'], \
                                                   num_workers = cfg['DEVICE']['NUM_WORKERS'], \
                                                   split_ratio = cfg['DATASET']['SPLIT_RATIO']) #Create train/validate data loader

        if (not main_loader): #Issue addressment
            return 
        
        [train_loader, val_loader] = main_loader #Train/Validate data loader
        
        ##### Step 7: Start model training/validation #####
        logger.info('\n')

        logger.info("*===================================================================*")
        logger.info("*================== Start training/validate model ==================*")
        logger.info("*===================================================================*")

        print('===================================================')

        main_trainer = Major_Loop_Trainer(cfg = cfg, logger = logger, num_classes = defect_classes) #Class object construct

        main_trainer.loop_trainer(train_loader = train_loader, val_loader = val_loader, model = main_model, \
                                  criterion = main_criterien, optimizer = main_optimizer, \
                                  scheduler = main_scheduler, device = main_device, fpath = target_fdir) #Train/Validate loop process/flow


    ##### Evaluation mode #####
    elif (cfg['eval']):
        target_fdir = join(cfg['OUTPUT'], 'evaluate_result') #Target path for evaluate

        if (not (exists(target_fdir))): #Create directory if path not exists
            mkdir(target_fdir)
        else:
            pass

        #########################
        #Temporal import section
        ##### Pytorch load weight/checkpoint #####
        from torch import load as torch_load

        ##### Evaluater #####
        from Evaluater import Major_Evaluater

        ##### Operating system (i.e. OS) #####
        from os.path import isdir, isfile


        ####################
        #Whole process/flow
        ##### Step 1: Dump/Save whole config/args information #####
        with open(join(target_fdir, 'config.yaml'), 'w') as f_dump: #Config yaml file
            dump(cfg, f_dump)

        ##### Step 2: Logging information file path/stream #####
        fh = FileHandler(join(target_fdir, 'log_info.log'), mode = 'w+') #Log info file
        fh.setFormatter(formatter)

        logger.addHandler(fh) #Add file handler to logger
        logger.addHandler(sh) #Add stream handler to logger

        logger.info('*************************************** Evaluate mode ***************************************') #Logger information

        for key in defect_dict:
            logger.info("|| {} : {}".format(key, (defect_dict[key] + 1))) #Logger information (i.e. defect class type)
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 3: GPU/CPU device #####
        if (is_available()): #Cuda available
            main_device = device('cuda:{}'.format(cfg['DEVICE']['GPU'])) #Point to GPU device

            logger.info('=> Using GPU device with cuda:{}'.format(cfg['DEVICE']['GPU']))

        else:
            main_device = device('cpu') #Point to CPU device

            logger.info('=> Using CPU device')

        logger.info("*-------------------------------------------------------------------*")

        ##### Step 4-1: Construct/Load model network/architecture #####
        if ((not cfg['MODEL']['BACKBONE']) or (not cfg['MODEL']['META_NETWORK']) \
            or (not cfg['eval_ckpt'])): #Issue addressment
            logger.error("Model informations can't be null !")

            return
        
        logger.info('=> Start constructing {} model: {} ...'.format(cfg['MODEL']['TASK'], cfg['MODEL']['META_NETWORK']))
        logger.info('==> Using backbone: {}'.format(cfg['MODEL']['BACKBONE']))

        main_model = Model_Mapper(cfg = cfg, num_classes = defect_classes) #Construct model architecture
        
        main_model = main_model.to(main_device) #Point model to main device

        Summarizer(logging = logger, model = main_model, device = main_device, \
                   channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                   height = cfg['INPUT']['RESOLUTION']['HEIGHT'], \
                   width = cfg['INPUT']['RESOLUTION']['WIDTH'], \
                   task = cfg['MODEL']['TASK']) #Summary for model
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 4-2: Load evaluation model weight/checkpoint #####
        logger.info("=> Using evaluate model checkpoint: {}".format(cfg['eval_ckpt']))
        
        ckpt = torch_load(cfg['eval_ckpt'], map_location = main_device) #Model checkpoint
        main_model.load_state_dict(ckpt['state_dict']) #Load checkpoint weight
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 5: Start model inference/evaluation #####
        if (not cfg['DATASET']['EVALUATE']): #Issue addressment
            logger.error("Evaluate set path can't be null !")

            return

        main_evaluater = Major_Evaluater(num_classes = defect_classes, camr_scl = cfg['INPUT']['CAMERA_SCALE'], \
                                         ref_scl = cfg['INPUT']['FILTERED_SCALE']['EVALUATE']['REFERENCE'], \
                                         post_scl = cfg['INPUT']['FILTERED_SCALE']['EVALUATE']['POST_PROCESS']) #Class object construct

        if (isfile(cfg['DATASET']['EVALUATE'])): #File/Dataset loader path
            from Evaluater import is_video #Temporal import file type/format (i.e. video, ...)

            if (not is_video(cfg['DATASET']['EVALUATE'])): #Dataset loader file name
                from Data.Dataset_Mapper import Dataloader_Mapper #Temporal import dataset loader
                from Loss.Loss_Function import Loss_Mapper #Temporal import loss function/criterion

                ##### Process/Flow 1: Loss function #####
                logger.info('=> Using loss function: {}'.format(cfg['LOSS']['FUNCTION']))

                main_criterien = Loss_Mapper(loss = cfg['LOSS']['FUNCTION'], \
                                             alpha = cfg['LOSS']['ALPHA'], \
                                             beta = cfg['LOSS']['BETA'], \
                                             gamma = cfg['LOSS']['GAMMA']) #Construct loss function
                
                main_criterien = main_criterien.to(main_device) #Point criterien function to main device
                
                logger.info("*-------------------------------------------------------------------*")

                ##### Process/Flow 2: Dataset loader #####
                main_mapper = Dataloader_Mapper(cfg = cfg, defect_dict = defect_dict) #Class object construct

                main_loader = main_mapper.evaluate_loader(logging = logger, task = cfg['MODEL']['TASK'], \
                                                          batch_size = cfg['INPUT']['BATCH_SIZE'], \
                                                          format = cfg['DATASET']['FORMAT'], \
                                                          num_workers = cfg['DEVICE']['NUM_WORKERS']) #Create evaluate data loader

                if (not main_loader): #Issue addressment
                    return 
            
                evaluate_loader = main_loader #Evaluate data loader

                ##### Process/Flow 3: Evaluater #####
                logger.info('\n')

                logger.info("*======================================================================*")
                logger.info("*================== Start evaluating/inference model ==================*")
                logger.info("*======================================================================*")

                print('===================================================')

                main_evaluater.dataloader_evaluater(eval_loader = evaluate_loader, model = main_model, \
                                                    criterion = main_criterien, device = main_device, \
                                                    logger = logger, task = cfg['MODEL']['TASK'], \
                                                    fpath = target_fdir) #Evaluate/Inference on dataset loader
                
            else: #Video-sreaming file name

                ##### Process/Flow 1: Evaluater #####
                logger.info('\n')
                logger.info("*======================================================================*")
                logger.info("*================== Start evaluating/inference model ==================*")
                logger.info("*======================================================================*")

                print('===================================================')

                main_evaluater.videowriter_evaluater(fpath = cfg['DATASET']['EVALUATE'], model = main_model, \
                                                     device = main_device, task = cfg['MODEL']['TASK'], \
                                                     target_fdir = target_fdir, logger = logger, \
                                                     defect_dict = defect_dict) #Evaluate/Inference on video-streaming file
            
        elif (isdir(cfg['DATASET']['EVALUATE'])): #Image folder directory path

            ##### Process/Flow 1: Evaluater #####
            logger.info('\n')
            logger.info("*======================================================================*")
            logger.info("*================== Start evaluating/inference model ==================*")
            logger.info("*======================================================================*")

            print('===================================================')

            main_evaluater.imagefolder_evaluater(fpath = cfg['DATASET']['EVALUATE'], model = main_model, \
                                                 device = main_device, task = cfg['MODEL']['TASK'], \
                                                 target_fdir = target_fdir, logger = logger, \
                                                 defect_dict = defect_dict) #Evaluate/Inference on image folder


    ##### Export mode #####
    elif (cfg['export']):
        target_fdir = join(cfg['OUTPUT'], 'export_result') #Target path for evaluate

        if (not (exists(target_fdir))): #Create directory if path not exists
            mkdir(target_fdir)
        else:
            pass

        #########################
        #Temporal import section
        ##### Pytorch load weight/checkpoint #####
        from torch import load as torch_load

        ##### Exporter #####
        from Exporter import Exporter_Mapper


        ####################
        #Whole process/flow
        ##### Step 1: Dump/Save whole config/args information #####
        with open(join(target_fdir, 'config.yaml'), 'w') as f_dump: #Config yaml file
            dump(cfg, f_dump)

        ##### Step 2: Logging information file path/stream #####
        fh = FileHandler(join(target_fdir, 'log_info.log'), mode = 'w+') #Log info file
        fh.setFormatter(formatter)

        logger.addHandler(fh) #Add file handler to logger
        logger.addHandler(sh) #Add stream handler to logger

        logger.info('*************************************** Export mode ***************************************') #Logger information

        for key in defect_dict:
            logger.info("|| {} : {}".format(key, (defect_dict[key] + 1))) #Logger information (i.e. defect class type)
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 3: GPU/CPU device #####
        if (is_available()): #Cuda available
            main_device = device('cuda:{}'.format(cfg['DEVICE']['GPU'])) #Point to GPU device

            logger.info('=> Using GPU device with cuda:{}'.format(cfg['DEVICE']['GPU']))

        else:
            main_device = device('cpu') #Point to CPU device

            logger.info('=> Using CPU device')

        logger.info("*-------------------------------------------------------------------*")

        ##### Step 4-1: Construct/Load model network/architecture #####
        if ((not cfg['MODEL']['BACKBONE']) or (not cfg['MODEL']['META_NETWORK']) \
             or (not cfg['export_ckpt'])): #Issue addressment
            logger.error("Model informations can't be null !")

            return
        
        logger.info('=> Start constructing {} model: {} ...'.format(cfg['MODEL']['TASK'], cfg['MODEL']['META_NETWORK']))
        logger.info('==> Using backbone: {}'.format(cfg['MODEL']['BACKBONE']))

        main_model = Model_Mapper(cfg = cfg, num_classes = defect_classes) #Construct model architecture
        
        main_model = main_model.to(main_device) #Point model to main device
        
        Summarizer(logging = logger, model = main_model, device = main_device, \
                   channel = cfg['INPUT']['RESOLUTION']['CHANNEL'], \
                   height = cfg['INPUT']['RESOLUTION']['HEIGHT'], \
                   width = cfg['INPUT']['RESOLUTION']['WIDTH'], \
                   task = cfg['MODEL']['TASK']) #Summary for model
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 4-2: Load export model weight/checkpoint #####
        logger.info("=> Using export model checkpoint: {}".format(cfg['export_ckpt']))
        
        ckpt = torch_load(cfg['export_ckpt'], map_location = main_device) #Model checkpoint
        main_model.load_state_dict(ckpt['state_dict']) #Load checkpoint weight
        
        logger.info("*-------------------------------------------------------------------*")

        ##### Step 5: Export pytorch model to format/file #####
        logger.info("=> Start exporting/convert model to {} format ...".format(cfg['EXPORT']['FORMAT']))

        Exporter_Mapper(cfg = cfg, logger = logger, fpath = target_fdir, model = main_model, \
                        device = main_device) #Export/Convert model to format



def Main(args: Any, logger: Any) -> None:
    #----------------------------------#
    # Description: Major/Main function #
    # Input type:                      #
    #   - Any (arguments)              #
    #   - Any (logging record)         #
    # Return type:                     #
    #   - None (void, no return)       #
    #----------------------------------#

    #########################
    #Main/Major process/flow
    Main_Worker(args = args, logger = logger)
    


#========================#
# Setup Code Entry Point #
#========================#
if (__name__ == "__main__"):
    Main(args = args, logger = logger) #Main/Major function call