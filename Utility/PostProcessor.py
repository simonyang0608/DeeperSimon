#***********************************************************************************************#
# Source: PostProcessor.py                                                                      #
#                                                                                               #
# Description: Post-process output results from model training/validation, inference/evaluation #
#                                                                                               #
# Author: SimonYang                                                                             #
#***********************************************************************************************#

#================#
# Import Section #
#================#
#############
#Numpy array
from numpy import array

##########
#Deepcopy
from copy import deepcopy

#############################
#Opencv (i.e. cv2) functions
from cv2 import (cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB, \
                 findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, \
                 boundingRect, rectangle, putText, \
                 FONT_HERSHEY_SIMPLEX, imwrite)


#======================#
# Define Function List #
#======================#
def Mask_To_Bbox_Multi(gt_dict: dict, filt_gt_dict: dict, pred_dict: dict, filt_pred_dict: dict, \
                       sub_argmax_mask: array, sub_target_mask: array, norm_ref_scl: list, \
                       norm_post_scl: list, num_classes: int) -> None:
    #------------------------------------------------------------------------#
    # Description: Convert mask to bounding-box format with multiple-classes #
    # Input type:                                                            #
    #   - dict (ground-truth dictionary)                                     #
    #   - dict (filtered ground-truth dictionary)                            #
    #   - dict (predict dictionary)                                          #
    #   - dict (filtered predict dictionary)                                 #
    #   - array (sub-mask with argument-max conversions)                     #
    #   - array (sub-target mask)                                            #
    #   - list (normalized filtered-scale list (i.e. reference))             #
    #   - list (normalized filtered-scale list (i.e. post-process))          #
    #   - int (number of labeled classes)                                    #
    # Return type:                                                           #
    #   - None (void, no return)                                             #
    #------------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Conversions between mask and bounding-box #####
    for classes_idx in range(num_classes):
        gt_binmask, pred_binmask = deepcopy(sub_target_mask), deepcopy(sub_argmax_mask) #Ground-truth, Predict mask initialize

        gt_binmask[(gt_binmask == classes_idx)] = 255 #Foreground pixel (i.e. ground-truth)
        gt_binmask[(gt_binmask != 255)] = 0 #Background pixel (i.e. ground-truth)

        pred_binmask[(pred_binmask == classes_idx)] = 255 #Foreground pixel (i.e. predict)
        pred_binmask[(pred_binmask != 255)] = 0 #Background pixel (i.e. predict)

        gt_contours, _ = findContours(gt_binmask.astype("uint8"), RETR_EXTERNAL, \
                                      CHAIN_APPROX_SIMPLE) #Extract contours from ground-truth mask
        
        pred_contours, _ = findContours(pred_binmask.astype("uint8"), RETR_EXTERNAL, \
                                        CHAIN_APPROX_SIMPLE) #Extract contours from predict mask
        
        gt_bboxes = [boundingRect(gt_cnt_position) for gt_cnt_position \
                     in gt_contours] #Get/Query each bounding box position (i.e. ground-truth)
        
        pred_bboxes = [boundingRect(pred_cnt_position) for pred_cnt_position \
                       in pred_contours] #Get/Query each bounding box position (i.e. predict)


        for gt_bbox in gt_bboxes:
            [offset_x, offset_y, width, height] = gt_bbox #Get/Query bounding rectangle/box parameters (i.e. ground-truth)

            gt_bbox = [offset_x, offset_y, (offset_x + width), (offset_y + height)] #Update bounding rectangle/box parameters (i.e. ground-truth)

            ##### Check if the current scales matched conditions or not #####
            if (((width <= norm_ref_scl[0][0]) and (height <= norm_ref_scl[0][1])) or \
                ((width <= norm_ref_scl[1][0]) and (height <= norm_ref_scl[1][1])) or \
                ((width <= norm_ref_scl[2][0]) and (height <= norm_ref_scl[2][1]))):
                pass

            else:
                (filt_gt_dict['boxes']).append(gt_bbox) #Bounding-boxes
                (filt_gt_dict['labels']).append((classes_idx + 1)) #Classes
                (filt_gt_dict['scores']).append(1.) #Scores

            (gt_dict['boxes']).append(gt_bbox) #Bounding-boxes
            (gt_dict['labels']).append((classes_idx + 1)) #Classes
            (gt_dict['scores']).append(1.) #Scores


        for pred_bbox in pred_bboxes:
            [offset_x, offset_y, width, height] = pred_bbox #Get/Query bounding rectangle/box parameters (i.e. predict)

            pred_bbox = [offset_x, offset_y, (offset_x + width), (offset_y + height)] #Update bounding rectangle/box parameters (i.e. predict)

            ##### Check if the current scales matched conditions or not #####
            if (((width <= norm_post_scl[0][0]) and (height <= norm_post_scl[0][1])) or \
                ((width <= norm_post_scl[1][0]) and (height <= norm_post_scl[1][1])) or \
                ((width <= norm_post_scl[2][0]) and (height <= norm_post_scl[2][1]))):
                pass

            else:
                (filt_pred_dict['boxes']).append(pred_bbox) #Bounding-boxes
                (filt_pred_dict['labels']).append((classes_idx + 1)) #Classes
                (filt_pred_dict['scores']).append(1.) #Scores

            (pred_dict['boxes']).append(pred_bbox) #Bounding-boxes
            (pred_dict['labels']).append((classes_idx + 1)) #Classes
            (pred_dict['scores']).append(1.) #Scores


        del gt_binmask #Delete/Release
        del pred_binmask #Delete/Release



def Mask_To_Bbox_Single(gt_dict: dict, filt_gt_dict: dict, pred_dict: dict, filt_pred_dict: dict, \
                        sub_gt_binmask: array, sub_pred_binmask: array, norm_ref_scl: list, \
                        norm_post_scl: list) -> None:
    #--------------------------------------------------------------------#
    # Description: Convert mask to bounding-box format with single-class #
    # Input type:                                                        #
    #   - dict (ground-truth dictionary)                                 #
    #   - dict (filtered ground-truth dictionary)                        #
    #   - dict (predict dictionary)                                      #
    #   - dict (filtered predict dictionary)                             #
    #   - array (sub-binary mask with ground-truths)                     #
    #   - array (sub-binary mask with predictions)                       #
    #   - list (normalized filtered-scale list (i.e. reference))         #
    #   - list (normalized filtered-scale list (i.e. post-process))      #
    # Return type:                                                       #
    #   - None (void, no return)                                         #
    #--------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Conversions between mask and bounding-box #####
    sub_gt_binmask[(sub_gt_binmask >= 1)] = 255 #Foreground pixel (i.e. ground-truth)
    sub_pred_binmask[(sub_pred_binmask >= 1)] = 255 #Foreground pixel (i.e. predict)

    gt_contours, _ = findContours(sub_gt_binmask.astype("uint8"), RETR_EXTERNAL, \
                                  CHAIN_APPROX_SIMPLE) #Extract contours from ground-truth mask
            
    pred_contours, _ = findContours(sub_pred_binmask.astype("uint8"), RETR_EXTERNAL, \
                                    CHAIN_APPROX_SIMPLE) #Extract contours from predict mask
    
    gt_bboxes = [boundingRect(gt_cnt_position) for gt_cnt_position \
                 in gt_contours] #Get/Query each bounding box position (i.e. ground-truth)
    
    pred_bboxes = [boundingRect(pred_cnt_position) for pred_cnt_position \
                   in pred_contours] #Get/Query each bounding box position (i.e. predict)


    for gt_bbox in gt_bboxes:
        [offset_x, offset_y, width, height] = gt_bbox #Get/Query bounding rectangle/box parameters (i.e. ground-truth)

        gt_bbox = [offset_x, offset_y, (offset_x + width), (offset_y + height)] #Update bounding rectangle/box parameters (i.e. ground-truth)

        ##### Check if the current scales matched conditions or not #####
        if (((width <= norm_ref_scl[0][0]) and (height <= norm_ref_scl[0][1])) or \
            ((width <= norm_ref_scl[1][0]) and (height <= norm_ref_scl[1][1])) or \
            ((width <= norm_ref_scl[2][0]) and (height <= norm_ref_scl[2][1]))):
            pass

        else:
            (filt_gt_dict['boxes']).append(gt_bbox) #Bounding-boxes
            (filt_gt_dict['labels']).append(1) #Classes
            (filt_gt_dict['scores']).append(1.) #Scores

        (gt_dict['boxes']).append(gt_bbox) #Bounding-boxes
        (gt_dict['labels']).append(1) #Classes
        (gt_dict['scores']).append(1.) #Scores


    for pred_bbox in pred_bboxes:
        [offset_x, offset_y, width, height] = pred_bbox #Get/Query bounding rectangle/box parameters (i.e. predict)

        pred_bbox = [offset_x, offset_y, (offset_x + width), (offset_y + height)] #Update bounding rectangle/box parameters (i.e. predict)

        ##### Check if the current scales matched conditions or not #####
        if (((width <= norm_post_scl[0][0]) and (height <= norm_post_scl[0][1])) or \
            ((width <= norm_post_scl[1][0]) and (height <= norm_post_scl[1][1])) or \
            ((width <= norm_post_scl[2][0]) and (height <= norm_post_scl[2][1]))):
            pass

        else:
            (filt_pred_dict['boxes']).append(pred_bbox) #Bounding-boxes
            (filt_pred_dict['labels']).append(1) #Classes
            (filt_pred_dict['scores']).append(1.) #Scores

        (pred_dict['boxes']).append(pred_bbox) #Bounding-boxes
        (pred_dict['labels']).append(1) #Classes
        (pred_dict['scores']).append(1.) #Scores



def Draw_Bbox_Det(len_pred_scores: int, iou_threshold: float, img_bgr: array, \
                  defect_list: list, color_arry: list, norm_post_scl: list, \
                  pred_scores: float, pred_boxes: array, pred_labels: int, \
                  show_filtered: bool) -> None:
    #---------------------------------------------------------------------#
    # Description: Draw-on bounding-boxes from models with detection task #
    # Input type:                                                         #
    #   - int (length of predict scores info.)                            #
    #   - float (self-defined iou-threshold)                              #
    #   - array (image data/info. in bgr format)                          #
    #   - list (self-defined defect list/categories)                      #
    #   - list (self-defined color list/categories)                       #
    #   - list (normalized filtered-scale list (i.e. post-process))       #
    #   - float (predict scores info.)                                    #
    #   - array (predict boxes info.)                                     #
    #   - float (predict labels info.)                                    #
    #   - bool (show filtered-defect flag)                                #
    # Return type:                                                        #
    #   - None (void, no return)                                          #
    #---------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Draw bounding-boxes on images #####
    for pred_idx in range(len_pred_scores):

        ##### Non-maximun suppression (i.e. NMS) operations #####
        if (pred_scores[pred_idx] > iou_threshold):
            min_x = int((pred_boxes[pred_idx])[0]) #Bbox minimun x co-ordinate
            min_y = int((pred_boxes[pred_idx])[1]) #Bbox minimun y co-ordinate
            max_x = int((pred_boxes[pred_idx])[2]) #Bbox maximun x co-ordinate
            max_y = int((pred_boxes[pred_idx])[3]) #Bbox maximun y co-ordinate

            x_scl, y_scl = (max_x - min_x), (max_y - min_y) #Bbox x, y length scales

            ##### Check if the current scales matched conditions or not #####
            if (((x_scl <= norm_post_scl[0][0]) and (y_scl <= norm_post_scl[0][1])) or \
                ((x_scl <= norm_post_scl[1][0]) and (y_scl <= norm_post_scl[1][1])) or \
                ((x_scl <= norm_post_scl[2][0]) and (y_scl <= norm_post_scl[2][1]))):

                ##### Check if it needs to show on filtered-defects or not #####
                if (show_filtered):
                    rectangle(img_bgr, (min_x, min_y), (max_x, max_y), \
                              (255, 255, 255), 2) #Draw matched bounding rectangle/boxes on image (i.e. filtered)
                    
                    putText(img_bgr, 'Filtered: 1.0', (min_x, (min_y - 8)), FONT_HERSHEY_SIMPLEX, \
                            0.5, (255, 255, 255), 2) #Assign matched defect class type on image (i.e. filtered)
                else:
                    pass

            else:
                rectangle(img_bgr, (min_x, min_y), (max_x, max_y), \
                          color_arry[(pred_labels[pred_idx] - 1)], 2) #Draw matched bounding rectangle/boxes on image
                
                putText(img_bgr, (defect_list[(pred_labels[pred_idx] - 1)] + ': ' + \
                        str(round(float(pred_scores[pred_idx]), 2))), (min_x, (min_y - 8)), \
                        FONT_HERSHEY_SIMPLEX, 0.5, color_arry[(pred_labels[pred_idx] - 1)], \
                        2) #Assign matched defect class type on image
            
        else:
            pass



def Draw_Bbox_Seg_Multi(argmax_mask: array, img_bgr: array, defect_list: list, \
                        color_arry: list, norm_post_scl: list, num_classes: int, \
                        show_filtered: bool) -> None:
    #------------------------------------------------------------------------------------------------#
    # Description: Draw-on bounding-boxes from models with segmentation task (i.e. multiple-classes) #
    # Input type:                                                                                    #
    #   - array (mask with argument-max conversions)                                                 #
    #   - array (image data/info. in bgr format)                                                     #
    #   - list (self-defined defect list/categories)                                                 #
    #   - list (self-defined color list/categories)                                                  #
    #   - list (normalized filtered-scale list (i.e. post-process))                                  #
    #   - int (number of labeled classes)                                                            #
    #   - bool (show filtered-defect flag)                                                           #
    # Return type:                                                                                   #
    #   - None (void, no return)                                                                     #
    #------------------------------------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Draw bounding-boxes on images #####
    for classes_idx in range(num_classes):
        pred_binmask = deepcopy(argmax_mask) #Predict binary mask initialize

        pred_binmask[(pred_binmask == classes_idx)] = 255 #Foreground pixel (i.e. predict)
        pred_binmask[(pred_binmask != 255)] = 0 #Background pixel (i.e. predict)

        contours, _ = findContours(((pred_binmask.astype("uint8"))), RETR_EXTERNAL, \
                                     CHAIN_APPROX_SIMPLE) #Extract contours from binary mask

        bboxes = [boundingRect(cnt_position) for cnt_position in contours] #Get/Query each bounding box position

        for bbox in bboxes:
            [offset_x, offset_y, width, height] = bbox #Get/Query bounding rectangle/box parameters

            ##### Check if the current scales matched conditions or not #####
            if (((width <= norm_post_scl[0][0]) and (height <= norm_post_scl[0][1])) or \
                ((width <= norm_post_scl[1][0]) and (height <= norm_post_scl[1][1])) or \
                ((width <= norm_post_scl[2][0]) and (height <= norm_post_scl[2][1]))):

                ##### Check if it needs to show on filtered-defects or not #####
                if (show_filtered):
                    rectangle(img_bgr, (offset_x, offset_y), ((offset_x + width), (offset_y + height)), \
                              (255, 255, 255), 2) #Draw matched bounding rectangle/boxes on image (i.e. filtered)
                    
                    putText(img_bgr, 'Filtered', (offset_x, (offset_y - 8)), FONT_HERSHEY_SIMPLEX, \
                            0.5, (255, 255, 255), 2) #Assign matched defect class type on image (i.e. filtered)
                else:
                    pass

            else:
                rectangle(img_bgr, (offset_x, offset_y), ((offset_x + width), (offset_y + height)), \
                          color_arry[classes_idx], 2) #Draw matched bounding rectangle/boxes on image
                
                putText(img_bgr, defect_list[classes_idx], (offset_x, (offset_y - 8)), \
                        FONT_HERSHEY_SIMPLEX, 0.5, color_arry[classes_idx], 2) #Assign matched defect class type on image

        del pred_binmask #Delete/Release
            


def Draw_Bbox_Seg_Single(binary_mask: array, img_bgr: array, defect_list: list, \
                         color_arry: list, norm_post_scl: list, show_filtered: bool) -> None:
    #--------------------------------------------------------------------------------------------#
    # Description: Draw-on bounding-boxes from models with segmentation task (i.e. single-class) #
    # Input type:                                                                                #
    #   - array (binary mask)                                                                    #
    #   - array (image data/info. in bgr format)                                                 #
    #   - list (self-defined defect list/categories)                                             #
    #   - list (self-defined color list/categories)                                              #
    #   - list (normalized filtered-scale list (i.e. post-process))                              #
    #   - bool (show filtered-defect flag)                                                       #
    # Return type:                                                                               #
    #   - None (void, no return)                                                                 #
    #--------------------------------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Draw bounding-boxes on images #####
    binary_mask[(binary_mask >= 1)] = 255 #Foreground pixel (i.e. binary mask)
                
    contours, _ = findContours(binary_mask, RETR_EXTERNAL, \
                               CHAIN_APPROX_SIMPLE) #Extract contours from thresholded mask

    bboxes = [boundingRect(cnt_position) for cnt_position in contours] #Get/Query each bounding box position

    for bbox in bboxes:
        [offset_x, offset_y, width, height] = bbox #Get/Query bounding rectangle/box parameters

        ##### Check if the current scales matched conditions or not #####
        if (((width <= norm_post_scl[0][0]) and (height <= norm_post_scl[0][1])) or \
            ((width <= norm_post_scl[1][0]) and (height <= norm_post_scl[1][1])) or \
            ((width <= norm_post_scl[2][0]) and (height <= norm_post_scl[2][1]))):

            ##### Check if it needs to show on filtered-defects or not #####
            if (show_filtered):
                rectangle(img_bgr, (offset_x, offset_y), ((offset_x + width), (offset_y + height)), \
                          (255, 255, 255), 2) #Draw matched bounding rectangle/boxes on image (i.e. filtered)
                
                putText(img_bgr, 'Filtered', (offset_x, (offset_y - 8)), FONT_HERSHEY_SIMPLEX, \
                        0.5, (255, 255, 255), 2) #Assign matched defect class type on image (i.e. filtered)
            else:
                pass

        else:
            rectangle(img_bgr, (offset_x, offset_y), ((offset_x + width), (offset_y + height)), \
                      color_arry[0], 2) #Draw matched bounding rectangle/boxes on image
            
            putText(img_bgr, defect_list[0], (offset_x, (offset_y - 8)), \
                    FONT_HERSHEY_SIMPLEX, 0.5, color_arry[0], 2) #Assign matched defect class type on image
        


def Filter_Target(norm_ref_scl: list, target: list, filt_target: list) -> None:
    #----------------------------------------------------------------------#
    # Description: Filter-out labeled-scales from ground-truth target list #                 
    # Input type:                                                          #
    #   - list (normalized filtered-scale list (i.e. reference))           #
    #   - list (labeled ground-truth target list)                          #
    #   - list (filtered-labeled ground-truth target list)                 #
    # Return type:                                                         #
    #   - None (void, no return)                                           #
    #----------------------------------------------------------------------#

    ############
    #Initialize
    ##### Length of target #####
    len_target = len(target)
    
    ####################
    #Whole process/flow
    ##### Step 1: Filter-out ground-truth bounding-boxes lower than defined-scale #####
    for target_idx in range(len_target):
        gt_bboxes = (target[target_idx])['boxes'] #Ground-truth bounding-boxes
        gt_labels = (target[target_idx])['labels'] #Ground-truth labels

        len_gt_bboxes = len(gt_bboxes) #Length of ground-truth bounding-boxes

        record_dict = {'boxes': [], 'labels': []} #Record dictionary/hashmap

        for bboxes_idx in range(len_gt_bboxes):
            [x1, y1, x2, y2] = gt_bboxes[bboxes_idx] #Get/Query bounding rectangle/box parameters

            width, height = (x2 - x1), (y2 - y1) #Get/Query bounding rectangle/box width, height

            ##### Check if the current scales matched conditions or not #####
            if (((width <= norm_ref_scl[0][0]) and (height <= norm_ref_scl[0][1])) or \
                ((width <= norm_ref_scl[1][0]) and (height <= norm_ref_scl[1][1])) or \
                ((width <= norm_ref_scl[2][0]) and (height <= norm_ref_scl[2][1]))):
                pass

            else:
                (record_dict['boxes']).append(gt_bboxes[bboxes_idx])
                (record_dict['labels']).append(gt_labels[bboxes_idx])

        filt_target.append(record_dict)



def Filter_Outputs(norm_post_scl: list, outputs: list, filt_outputs: list) -> None:
    #-----------------------------------------------------------------------------#
    # Description: Filter-out labeled-scales from predicted/inference target list #                 
    # Input type:                                                                 #
    #   - list (normalized filtered-scale list (i.e. post-process))               #
    #   - list (predicted/inferenced outputs list)                                #
    #   - list (filtered-predicted/inferenced outputs list)                       #
    # Return type:                                                                #
    #   - None (void, no return)                                                  #
    #-----------------------------------------------------------------------------#
    
    ############
    #Initialize
    ##### Length of outputs #####
    len_outputs = len(outputs)

    ####################
    #Whole process/flow
    ##### Step 1: Filter-out predict bounding-boxes lower than defined-scale #####
    for outputs_idx in range(len_outputs):
        pred_bboxes = (outputs[outputs_idx])['boxes'] #Predict bounding-boxes
        pred_labels = (outputs[outputs_idx])['labels'] #Predict labels
        pred_scores = (outputs[outputs_idx])['scores'] #Predict scores

        len_pred_bboxes = len(pred_bboxes) #Length of predict bounding-boxes

        record_dict = {'boxes': [], 'labels': [], 'scores':[]} #Record dictionary/hashmap

        for bboxes_idx in range(len_pred_bboxes):
            [x1, y1, x2, y2] = pred_bboxes[bboxes_idx] #Get/Query bounding rectangle/box parameters

            width, height = (x2 - x1), (y2 - y1) #Get/Query bounding rectangle/box width, height

            ##### Check if the current scales matched conditions or not #####
            if (((width <= norm_post_scl[0][0]) and (height <= norm_post_scl[0][1])) or \
                ((width <= norm_post_scl[1][0]) and (height <= norm_post_scl[1][1])) or \
                ((width <= norm_post_scl[2][0]) and (height <= norm_post_scl[2][1]))):
                pass

            else:
                (record_dict['boxes']).append(pred_bboxes[bboxes_idx])
                (record_dict['labels']).append(pred_labels[bboxes_idx])
                (record_dict['scores']).append(pred_scores[bboxes_idx])

        filt_outputs.append(record_dict)