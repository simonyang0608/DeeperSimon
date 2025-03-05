#*********************************************************************#
# Source: PreProcessor.py                                             #
#                                                                     #
# Description: Customized self-defined dataset format preprocess/flow #
#                                                                     #
# Author: SimonYang                                                   #
#*********************************************************************#

#================#
# Import Section #
#================#
#############################################
#Pytorch datasets, transforms/rest functions
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import argmax, from_numpy

##### Random times/values #####
from random import randint

##############
#Pillow image
from PIL.Image import open

#################
#Numpy functions
from numpy import (zeros, ones, array)

####################
#Typing format list
from typing import Any, List


#=====================#
# Class Function List #
#=====================#
class COCO_PreProcessor(Dataset):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, fpath: str, task: str, transform_list: List[dict], height: int, \
                 width: int, defect_dict: dict) -> None:
        #---------------------------------------------------------#
        # Description: Constructor initialize/setup               #
        # Input type:                                             #
        #   - str (file full path)                                #
        #   - str (tasks type)                                    #
        #   - List[dict] (augmentation transform dictionary list) #
        #   - int (image height)                                  #
        #   - int (image width)                                   #
        #   - dict (defect defined dictionary)                    #
        # Return type:                                            #
        #   - None (void, no return)                              #
        #---------------------------------------------------------#
        from pycocotools.coco import COCO #Temporal import COCO

        ############
        #Initialize
        ##### COCO loader, ids #####
        self.coco = COCO(fpath)
        self.ids = list(sorted(((self.coco).imgs).keys()))

        ##### Task type #####
        self.task = task

        ##### Augmentation transform, length #####
        self.transform_list, self.len_transform_list = transform_list, len(transform_list)

        ##### Image heigth, width #####
        self.height, self.width = height, width

        ##### Defect dictionary #####
        self.defect_dict = defect_dict
        self.defect_len = len(self.defect_dict)

        ##### Masks #####
        self.init_mask = [zeros((height, width)) for _ in range(self.defect_len)]

        if (self.defect_len > 1):
            self.init_mask.append(ones((height, width))) #Background class

            self.defect_len += 1

        else:
            pass

        self.init_mask = array(self.init_mask)


    ########################
    # Member Function List #
    ########################
    def __getitem__(self, index: int) -> Any:
        #--------------------------------------------------------#
        # Description: Get/Query whole items with specific index #
        # Input type:                                            #
        #   - int (specific index)                               #
        # Return type:                                           #
        #   - Any (return any type)                              #
        #--------------------------------------------------------#

        ############
        #Initialize
        ##### COCO #####
        coco = self.coco

        ##### Get/Query COCO annotations from image id #####
        img_ids = (self.ids)[index]
        ann_ids = coco.getAnnIds(img_ids)
        ann_ids_item = coco.loadAnns(ann_ids)

        ##### Number of objects from COCO image id annotations #####
        num_objs = len(ann_ids_item)

        ##### Task #####
        task = self.task

        ##### Random times/values #####
        rand_val = randint(0, (self.len_transform_list - 1))

        ##### Transform #####
        transform = (self.transform_list)[rand_val]

        ##### Image heigth, width #####
        height, width = self.height, self.width

        ##### Defect dictionary, class length #####
        defect_dict = self.defect_dict
        defect_len = self.defect_len

        ##### COCO Image info. #####
        img_info = (coco.loadImgs(img_ids))[0]
        img = open(img_info['file_name'])

        if (img.mode != 'RGB'):
            img = img.convert('RGB')

        else:
            pass


        ########################################
        #Customized process for different tasks
        ##### Detection #####
        if (task == 'detection'):
            init_bbox, init_cls_idx = [], [] #Bounding-boxes, classes initialize
            target = {} #Target initialize
            
            for obj_idx in range(num_objs):

                ##### Check if defect defined to be detect or not #####
                if (ann_ids_item[obj_idx]['subcategory'] in defect_dict):
                    (init_bbox).append(array(ann_ids_item[obj_idx]['bbox'])) #Bounding boxes append/accumulate
                    (init_cls_idx).append(defect_dict[ann_ids_item[obj_idx]['subcategory']]) #Class indexes append/accumulate

                else:
                    pass

            ##### Check if images/objects needs to be detected or not #####
            if (init_bbox and init_cls_idx):
                final_bbox, final_cls_idx = array(init_bbox), array(init_cls_idx) #Final convert bounding boxes, class indexes to input loader shape format (i.e. ... x C)

                bbox = from_numpy(final_bbox) #Bounding boxes conversions

                bbox[:, [2]] = (bbox[:, [0]] + bbox[:, [2]]) #Convert bounding boxes to (x1, y1, x2, y2) format
                bbox[:, [3]] = (bbox[:, [1]] + bbox[:, [3]]) #Convert bounding boxes to (x1, y1, x2, y2) format
                
                image = transform['image'](img) #Image transformations

                ##### Check if suitable for bounding boxes transformations or not #####
                if ('bbox' in transform):

                    ##### Check which transformations type #####
                    if (transform['bbox'] == 'h_flip'):
                        bbox[:, [0, 2]] = (width - bbox[:, [2, 0]]) #Horizontal-flip

                    elif (transform['bbox'] == 'v_flip'):
                        bbox[:, [1, 3]] = (height - bbox[:, [3, 1]]) #Vertical-flip

                    elif (transform['bbox'] == 'both_flip'):
                        bbox[:, [0, 2]] = (width - bbox[:, [2, 0]]) #Horizontal-flip
                        bbox[:, [1, 3]] = (height - bbox[:, [3, 1]]) #Vertical-flip

                else:
                    pass

                cls_idx = from_numpy(final_cls_idx) #Class indexes conversions

                target['boxes'] = bbox #Bounding boxes records/updates
                target['labels'] = (cls_idx + 1) #Class indexes records/updates
            
            else:
                final_bbox, final_cls_idx = (array(init_bbox).reshape(-1, 4)).astype('float32'), \
                    array(init_cls_idx).astype('int64') #Final convert bounding boxes, class indexes to input loader shape format (i.e. ... x C)

                image = transform['image'](img) #Image transformations

                target['boxes'] = from_numpy(final_bbox) #Bounding boxes records/updates
                target['labels'] = from_numpy(final_cls_idx) #Class indexes records/updates

            return image, target


        ##### Segmentation #####
        elif (task == 'segmentation'):

            ##### Check if one or multiple channels (i.e. background class exist)
            if (defect_len > 1):
                self.init_mask[: (defect_len - 1)][self.init_mask[: (defect_len - 1)] != 0] = 0 #Reset
                self.init_mask[(defect_len - 1)][self.init_mask[(defect_len - 1)] != 1] = 1 #Reset
            else:
                self.init_mask[(defect_len - 1)][self.init_mask[(defect_len - 1)] != 0] = 0 #Reset

            ##### Check it's PASS or NG #####
            if (not ann_ids_item):
                pass
            
            else:
                for mask_idx in range(num_objs):

                    ##### Check if defect defined to be detect or not #####
                    if (ann_ids_item[mask_idx]['subcategory'] in defect_dict):
                        self.init_mask[defect_dict[ann_ids_item[mask_idx]['subcategory']]] += coco.annToMask(ann_ids_item[mask_idx]) #Remained mask instances accumulate

                        ##### Check if one or multiple channels (i.e. background class exist)
                        if (defect_len > 1):
                            self.init_mask[-1] -= coco.annToMask(ann_ids_item[mask_idx]) #Subtract mask from background
                        else:
                            pass

                    else:
                        pass
            
            final_mask = ((self.init_mask * 255).transpose([1, 2, 0])).astype('uint8') #Final convert mask to input loader shape format (i.e. H x W x C)

            image = transform['image'](img) #Image transformations
            mask = transform['mask'](final_mask) #Mask transformations

            ##### Check if one or multiple channels (i.e final transformations) #####
            if (defect_len > 1):
                mask = argmax(mask, dim = 0) #Argument-max conversions
            else:
                pass

            return image, mask


        ##### Classification #####
        elif (task == 'classification'):
            pass


    def __len__(self) -> int:
        #-------------------------------------------#
        # Description: Get/Query lengh of whole ids #
        # Input type:                               #
        #   - None (void)                           #
        # Return type:                              #
        #   - int (result lengh of whole ids)       #
        #-------------------------------------------#

        return (len(self.ids))



#=================#
# Mapper Function #
#=================#
def PreProcessor_Mapper(format: str, fpath: str, task: str, transform_list: List[dict], \
                        height: int, width: int, defect_dict: dict) -> Dataset:
    #-------------------------------------------------------------#
    # Description: Customized mapper for self-defined data format #
    # Input type:                                                 #
    #   - str (dataset format)                                    #
    #   - str (file full path)                                    #
    #   - str (tasks type)                                        #
    #   - List[dict] (augmentation transform dictionary list)     #
    #   - int (image height)                                      #
    #   - int (image width)                                       #
    #   - dict (defect defined dictionary)                        #
    # Return type:                                                #
    #   - Dataset (result dataset)                                #
    #-------------------------------------------------------------#

    ############
    #Initialize
    ##### Mapper hashmap/dictionary #####
    mapper_dict = {}

    ##############################################
    #Mapper process with different pre-processors
    ##### COCO #####
    mapper_dict['coco'] = COCO_PreProcessor(fpath = fpath, task = task, transform_list = transform_list, \
                                            height = height, width = width, defect_dict = defect_dict)

    return mapper_dict[format]