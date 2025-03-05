#*********************************************************#
# Source: Dataset_Mapper.py                               #
#                                                         #
# Description: Dataloader mapper for self-defined dataset #
#                                                         #
# Author: SimonYang                                       #
#*********************************************************#

#================#
# Import Section #
#================#
###############################################################
#PreProcessor mapper, typing format list, transforms functions
from Data.PreProcessor import (PreProcessor_Mapper, Any, List, ToTensor)

#######################################
#Pytorch dataloader, dataset functions
from torch.utils.data import (DataLoader, random_split, ConcatDataset)

########################
#Torchvision transforms
from torchvision.transforms import (RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, \
                                    Compose)


#=====================#
# Class Function List #
#=====================#
class Dataloader_Mapper(object):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, cfg: dict, defect_dict: dict) -> None:
        #-------------------------------------------#
        # Description: Constructor initialize/setup #
        # Input type:                               #
        #   - dict (whole config information)       #
        #   - dict (defect defined dictionary)      #
        # Return type:                              #
        #   - None (void, no return)                #
        #-------------------------------------------#

        ############
        #Initialize
        ##### NG/PASS, evaluate dataset path #####
        self.ng_path = cfg['DATASET']['TRAIN_VAL']['NG']
        self.pass_path = cfg['DATASET']['TRAIN_VAL']['PASS']
        self.eval_path = cfg['DATASET']['EVALUATE']

        ##### Image height, width #####
        self.height = cfg['INPUT']['RESOLUTION']['HEIGHT']
        self.width = cfg['INPUT']['RESOLUTION']['WIDTH']

        ##### Defect dictionary #####
        self.defect_dict = defect_dict

        ##### Base, Augment transform list (i.e. (image, mask, bbox, ...)) #####
        self.base_transform_list, self.transform_list = [{'image': Compose([ToTensor()]), \
                                                          'mask': Compose([ToTensor()])}], []
        
        self.flip_horizontal = cfg['DATASET']['TRANSFORM']['FLIP']['HORIZONTAL'] #Horizontal-flip augment list
        self.flip_vertical = cfg['DATASET']['TRANSFORM']['FLIP']['VERTICAL'] #Vertical-flip augment list

        if (self.flip_horizontal and self.flip_vertical):
            (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                               'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                               'bbox': 'both_flip'})
        else:
            if (self.flip_horizontal):
                (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                   'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                   'bbox': 'h_flip'})
            elif (self.flip_vertical):
                (self.base_transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                   'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                   'bbox': 'v_flip'})
            else:
                pass

        self.colorjitter_hue = cfg['DATASET']['TRANSFORM']['COLOR']['HUE'] #Hue augment list
        self.colorjitter_contrast = cfg['DATASET']['TRANSFORM']['COLOR']['CONTRAST'] #Contrast augment list
        self.colorjitter_saturation = cfg['DATASET']['TRANSFORM']['COLOR']['SATURATION'] #Saturation augment list
        self.colorjitter_brightness = cfg['DATASET']['TRANSFORM']['COLOR']['BRIGHTNESS'] #Brightness augment list


        for self.hue in self.colorjitter_hue: #Hue augment
            (self.transform_list).append({'image': Compose([ToTensor(), ColorJitter(hue = (self.hue, self.hue))]), \
                                          'mask': Compose([ToTensor()])})
            (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(hue = (self.hue, self.hue))]), \
                                               'mask': Compose([ToTensor()])})
            
            if (self.flip_horizontal and self.flip_vertical):
                (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                              'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                              'bbox': 'both_flip'})
                (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                                   'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                   'bbox': 'both_flip'})
            else:
                if (self.flip_horizontal):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                                  'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                  'bbox': 'h_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                       'bbox': 'h_flip'})
                elif (self.flip_vertical):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                                  'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                  'bbox': 'v_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(hue = (self.hue, self.hue))]), \
                                                       'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'v_flip'})
                else:
                    pass
            
            for self.brightness in self.colorjitter_brightness: #Brightness augment
                (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                     ColorJitter(hue = (self.hue, self.hue))]), \
                                                   'mask': Compose([ToTensor()])})
                
                if (self.flip_horizontal and self.flip_vertical):
                    (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                     ColorJitter(hue = (self.hue, self.hue)), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'both_flip'})
                else:
                    if (self.flip_horizontal):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                     ColorJitter(hue = (self.hue, self.hue)), RandomHorizontalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                           'bbox': 'h_flip'})
                    elif (self.flip_vertical):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                     ColorJitter(hue = (self.hue, self.hue)), RandomVerticalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                           'bbox': 'v_flip'})
                    else:
                        pass


        for self.contrast in self.colorjitter_contrast: #Contrast augment
            (self.transform_list).append({'image': Compose([ToTensor(), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                          'mask': Compose([ToTensor()])})
            (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                               'mask': Compose([ToTensor()])})

            if (self.flip_horizontal and self.flip_vertical):
                (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                              'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                              'bbox': 'both_flip'})
                (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                   'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                   'bbox': 'both_flip'})
            else:
                if (self.flip_horizontal):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                  'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                  'bbox': 'h_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                       'bbox': 'h_flip'})
                elif (self.flip_vertical):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                  'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                  'bbox': 'v_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                       'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'v_flip'})
                else:
                    pass
            
            for self.brightness in self.colorjitter_brightness: #Brightness augment
                (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                     ColorJitter(contrast = (self.contrast, self.contrast))]), \
                                                   'mask': Compose([ToTensor()])})
                                                   
                if (self.flip_horizontal and self.flip_vertical):
                    (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                     ColorJitter(contrast = (self.contrast, self.contrast)), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'both_flip'})
                else:
                    if (self.flip_horizontal):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                            ColorJitter(contrast = (self.contrast, self.contrast)), RandomHorizontalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                           'bbox': 'h_flip'})
                    elif (self.flip_vertical):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                            ColorJitter(contrast = (self.contrast, self.contrast)), RandomVerticalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                           'bbox': 'v_flip'})
                    else:
                        pass
            

        for self.saturation in self.colorjitter_saturation: #Saturation augment
            (self.transform_list).append({'image': Compose([ToTensor(), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                          'mask': Compose([ToTensor()])})
            (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                               'mask': Compose([ToTensor()])})

            if (self.flip_horizontal and self.flip_vertical):
                (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                              'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                              'bbox': 'both_flip'})
                (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                   'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                   'bbox': 'both_flip'})
            else:
                if (self.flip_horizontal):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                'bbox': 'h_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomHorizontalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                       'bbox': 'h_flip'})
                elif (self.flip_vertical):
                    (self.transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                  'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                  'bbox': 'v_flip'})
                    (self.base_transform_list).append({'image': Compose([ToTensor(), RandomVerticalFlip(p = 1), ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                       'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'v_flip'})
                else:
                    pass
            
            for self.brightness in self.colorjitter_brightness: #Brightness augment
                (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                     ColorJitter(saturation = (self.saturation, self.saturation))]), \
                                                   'mask': Compose([ToTensor()])}), \
                                                   
                if (self.flip_horizontal and self.flip_vertical):
                    (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                     ColorJitter(saturation = (self.saturation, self.saturation)), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'both_flip'})

                else:
                    if (self.flip_horizontal):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                            ColorJitter(saturation = (self.saturation, self.saturation)), RandomHorizontalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                           'bbox': 'h_flip'})
                    elif (self.flip_vertical):
                        (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)),\
                                                                            ColorJitter(saturation = (self.saturation, self.saturation)), RandomVerticalFlip(p = 1)]), \
                                                           'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                           'bbox': 'v_flip'})
                    else:
                        pass
            

        for self.brightness in self.colorjitter_brightness: #Brightness augment
            (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness))]), \
                                               'mask': Compose([ToTensor()])}), \
                                             
            if (self.flip_horizontal and self.flip_vertical):
                (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                 RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                   'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1), RandomVerticalFlip(p = 1)]), \
                                                   'bbox': 'both_flip'})
            else:
                if (self.flip_horizontal):
                    (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                        RandomHorizontalFlip(p = 1)]), \
                                                       'mask': Compose([ToTensor(), RandomHorizontalFlip(p = 1)]), \
                                                       'bbox': 'h_flip'})
                elif (self.flip_vertical):
                    (self.base_transform_list).append({'image': Compose([ToTensor(), ColorJitter(brightness = (self.brightness, self.brightness)), \
                                                                        RandomVerticalFlip(p = 1)]), \
                                                       'mask': Compose([ToTensor(), RandomVerticalFlip(p = 1)]), \
                                                       'bbox': 'v_flip'})
                else:
                    pass
            

    ########################
    # Member Function List #
    ########################
    def train_val_loader(self, logging: Any, task: str, batch_size: int, \
                         format: str, num_workers: int, split_ratio: float) -> List[DataLoader]:
        #-----------------------------------------------#
        # Description: Train/Valid dataset loader       #
        # Input type:                                   #
        #   - Any (logging record)                      #
        #   - str (tasks type)                          #
        #   - int (batch size)                          #
        #   - str (dataset format)                      #
        #   - int (number of workers)                   #
        #   - float (split ratio)                       #
        # Return type:                                  #
        #   - List[DataLoader] (result dataloader list) #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### NG/PASS path #####
        ng_path = self.ng_path
        pass_path = self.pass_path

        ##### Image height, width #####
        height = self.height
        width = self.width

        ##### Defect dictionary #####
        defect_dict = self.defect_dict

        ##### Base, Augment transform list #####
        base_transform_list = self.base_transform_list
        transform_list = self.transform_list


        ##################################
        #Generate train, valid dataloader
        ##### Check if valid input dataset path/samples #####
        if ((not ng_path) and (not pass_path)):
            logging.error("Dataset path can't be null !")

            return False
        
        else:
            logging.info('=> Start loading train/validate dataset ...') #Start loading information

            ##### Check the NG/PASS dataset distribution #####
            if (ng_path):
                ng_set_init = PreProcessor_Mapper(format = format, fpath = ng_path, task = task, \
                                                  transform_list = base_transform_list, height = height,\
                                                  width = width, defect_dict = defect_dict) #NG set initialize

                ng_set_len = len(ng_set_init) #Get/Query length of NG set
                
                ##### Check if both PASS/NG dataset exist #####
                if (pass_path):
                    pass_set_init = PreProcessor_Mapper(format = format, fpath = pass_path, task = task, \
                                                        transform_list = base_transform_list, height = height, \
                                                        width = width, defect_dict = defect_dict) #PASS set initialize              
                    
                    pass_set_len = len(pass_set_init) #Get/Query length of PASS set

                    ##### Check the samples quantity of PASS/NG #####
                    if (ng_set_len > pass_set_len):
                        logging.info('=> Augments {} times for PASS samples with balanced ...'.format(round(ng_set_len/pass_set_len) - 1)) #Augment information

                        augment_pass_set = [PreProcessor_Mapper(format = format, fpath = pass_path, task = task, \
                                                                transform_list = [transform_list[transforms_idx]], \
                                                                height = height, width = width, defect_dict = defect_dict) \
                                                                for transforms_idx in range(min(((round(ng_set_len/pass_set_len) - 1)), len(transform_list)))] #Proceed augmentation process
                        
                        total_pass_set = ConcatDataset(augment_pass_set + [pass_set_init]) #Concate dataset

                        logging.info('==> Total with {} PASS samples'.format(len(total_pass_set))) #Number of PASS samples
                        logging.info('==> Total with {} NG samples'.format(ng_set_len)) #Number of NG samples

                        train_pass, val_pass = random_split(total_pass_set, [(len(total_pass_set) - int(len(total_pass_set) * split_ratio)), \
                                                                              int(len(total_pass_set) * split_ratio)]) #Split PASS into train/val
                        
                        train_ng, val_ng = random_split(ng_set_init, [(ng_set_len - int(ng_set_len * split_ratio)), \
                                                                       int(ng_set_len * split_ratio)]) #Split NG into train/val

                    else:
                        logging.info('=> Augments {} times for NG samples with balanced ...'.format(round(pass_set_len/ng_set_len) - 1)) #Augment information

                        augment_ng_set = [PreProcessor_Mapper(format = format, fpath = ng_path, task = task, \
                                                              transform_list = [transform_list[transforms_idx]], \
                                                              height = height, width = width, defect_dict = defect_dict) \
                                                              for transforms_idx in range(min((round(pass_set_len/ng_set_len) - 1), len(transform_list)))] #Proceed augmentation process
                        
                        total_ng_set = ConcatDataset(augment_ng_set + [ng_set_init]) #Concate dataset

                        logging.info('==> Total with {} PASS samples'.format(pass_set_len)) #Number of PASS samples
                        logging.info('==> Total with {} NG samples'.format(len(total_ng_set))) #Number of NG samples

                        train_pass, val_pass = random_split(pass_set_init, [(pass_set_len - int(pass_set_len * split_ratio)), \
                                                                             int(pass_set_len * split_ratio)]) #Split PASS into train/val
                        
                        train_ng, val_ng = random_split(total_ng_set, [(len(total_ng_set) - int(len(total_ng_set) * split_ratio)), \
                                                                        int(len(total_ng_set) * split_ratio)]) #Split NG into train/val
                        
                    train_pass_ng, val_pass_ng = ConcatDataset([train_pass, train_ng]), ConcatDataset([val_pass, val_ng]) #Combine PASS/NG into train/val

                    logging.info('===> Train on {} PASS/NG samples'.format(round(len(train_pass_ng)))) #Number of train PASS/NG samples
                    logging.info('===> Validate on {} PASS/NG samples'.format(round(len(val_pass_ng)))) #Number of validate PASS/NG samples
                        
                    ##### Customized dataloader status with different tasks #####
                    if (task == 'detection'): #Detection

                        return [DataLoader(dataset = ConcatDataset([train_pass_ng, val_pass_ng]), batch_size = batch_size, shuffle = True, \
                                           num_workers = num_workers, pin_memory = True, collate_fn = self.collate_fn), \
                                DataLoader(dataset = val_pass_ng, batch_size = batch_size, shuffle = False, \
                                           num_workers = num_workers, pin_memory = True, \
                                           collate_fn = self.collate_fn)]
                    
                    elif (task == 'segmentation'): #Segmentation
                        
                        return [DataLoader(dataset = ConcatDataset([train_pass_ng, val_pass_ng]), batch_size = batch_size, shuffle = True, \
                                           num_workers = num_workers, pin_memory = True), \
                                DataLoader(dataset = val_pass_ng, batch_size = batch_size, shuffle = False, \
                                           num_workers = num_workers, pin_memory = True)]
                        
                else:
                    logging.info('==> Total with {} NG samples'.format(ng_set_len)) #Number of NG samples

                    train_ng, val_ng = random_split(ng_set_init, [(ng_set_len - int(ng_set_len * split_ratio)), \
                                                                   int(ng_set_len * split_ratio)]) #Split NG into train/val
                    
                    logging.info('===> Train on {} NG samples'.format(round(len(train_ng)))) #Number of train NG samples
                    logging.info('===> Validate on {} NG samples'.format(round(len(val_ng)))) #Number of validate NG samples
                    
                    ##### Customized dataloader status with different tasks #####
                    if (task == 'detection'): #Detection

                        return [DataLoader(dataset = ConcatDataset([train_ng, val_ng]), batch_size = batch_size, shuffle = True, \
                                           num_workers = num_workers, pin_memory = True, collate_fn = self.collate_fn), \
                                DataLoader(dataset = val_ng, batch_size = batch_size, shuffle = False, \
                                           num_workers = num_workers, pin_memory = True, \
                                           collate_fn = self.collate_fn)]
                    
                    elif (task == 'segmentation'): #Segmentation

                        return [DataLoader(dataset = ConcatDataset([train_ng, val_ng]), batch_size = batch_size, shuffle = True, \
                                           num_workers = num_workers, pin_memory = True), \
                                DataLoader(dataset = val_ng, batch_size = batch_size, shuffle = False, \
                                           num_workers = num_workers, pin_memory = True)]
                
            else:
                pass_set_init = PreProcessor_Mapper(format = format, fpath = pass_path, task = task, \
                                                    transform_list = base_transform_list, height = height, \
                                                    width = width, defect_dict = defect_dict) #PASS set initialize

                pass_set_len = len(pass_set_init) #Get/Query length of PASS set         
                
                logging.info('==> Total with {} PASS samples'.format(pass_set_len)) #Number of PASS samples
                
                train_pass, val_pass = random_split(pass_set_init, [(pass_set_len - int(pass_set_len * split_ratio)), \
                                                                     int(pass_set_len * split_ratio)]) #Split PASS into train/val
                
                logging.info('===> Train on {} PASS samples'.format(round(len(train_pass)))) #Number of train PASS samples
                logging.info('===> Validate on {} PASS samples'.format(round(len(val_pass)))) #Number of validate PASS samples
                
                ##### Customized dataloader status with different tasks #####
                if (task == 'detection'): #Detection

                    return [DataLoader(dataset = ConcatDataset([train_pass, val_pass]), batch_size = batch_size, shuffle = True, \
                                       num_workers = num_workers, pin_memory = True, collate_fn = self.collate_fn), \
                            DataLoader(dataset = val_pass, batch_size = batch_size, shuffle = False, \
                                       num_workers = num_workers, pin_memory = True, \
                                       collate_fn = self.collate_fn)]
                
                elif (task == 'segmentation'): #Segmentation

                    return [DataLoader(dataset = ConcatDataset([train_pass, val_pass]), batch_size = batch_size, shuffle = True, \
                                       num_workers = num_workers, pin_memory = True), \
                            DataLoader(dataset = val_pass, batch_size = batch_size, shuffle = False, \
                                       num_workers = num_workers, pin_memory = True)]


    def evaluate_loader(self, logging: Any, task: str, batch_size: int, \
                        format: str, num_workers: int) -> DataLoader:
        #------------------------------------- #
        # Description: Evaluate dataset loader #
        # Input type:                          #
        #   - Any (logging record)             #
        #   - str (tasks type)                 #
        #   - int (batch size)                 #
        #   - str (dataset format)             #
        #   - int (number of workers)          #
        # Return type:                         #
        #   - Dataloader (result dataloader)   #
        #--------------------------------------#

        ############
        #Initialize
        ##### Evaluate path #####
        eval_path = self.eval_path

        ##### Image height, width #####
        height = self.height
        width = self.width

        ##### Defect dictionary #####
        defect_dict = self.defect_dict


        ##############################
        #Generate evaluate dataloader
        ##### Check if valid input dataset path #####
        if (not eval_path):
            logging.error("Dataset path can't be null !")

            return False
        
        else:
            logging.info('=> Start loading evaluate dataset ...') #Start loading information

            eval_set = PreProcessor_Mapper(format = format, fpath = eval_path, task = task, \
                                           transform_list = [{'image': Compose([ToTensor()]), \
                                                              'mask': Compose([ToTensor()])}], \
                                           height = height, width = width, \
                                           defect_dict = defect_dict) #Evaluate set
            
            logging.info('==> Evaluate on {} samples'.format(len(eval_set))) #Get/Query length information

            ##### Customized dataloader status with different tasks #####
            if (task == 'detection'): #Detection

                return DataLoader(dataset = eval_set, batch_size = batch_size, shuffle = False, \
                                  num_workers = num_workers, pin_memory = True, \
                                  collate_fn = self.collate_fn)
            
            elif (task == 'segmentation'): #Segmentation

                return DataLoader(dataset = eval_set, batch_size = batch_size, shuffle = False, \
                                  num_workers = num_workers, pin_memory = True)
            

    def collate_fn(self, batch_sample: int) -> Any:
        #--------------------------------------------------#
        # Description: Collate function filtered/processed #
        # Input type:                                      #
        #   - int (batch dataset sample)                   #
        # Return type:                                     #
        #   - Any (result dataset structure)               #
        #--------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Filtered batch data samples which returned None type #####
        batch_sample = list(filter(lambda x: x is not None, batch_sample))

        ##### Step 2: Zipped-matches for bounding-boxes and labels #####
        batch_sample = list(zip(*batch_sample))

        return batch_sample