#****************************************************************************#
# Source: Metric.py                                                          #
#                                                                            #
# Description: Measured output results metric from model training/validation #
#                                                                            #
# Author: SimonYang                                                          #
#****************************************************************************#

#================#
# Import Section #
#================#
##############################
#Pytorch numel, Tensor, stack
from torch import (numel, Tensor)

#############################
#Numpy array, rest functions
from numpy import array

###############################
#Scikit-learn metric functions
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix)

####################
#Typing format list
from typing import Any, List


#======================#
# Define Function List #
#======================#
def Binary_Pixel_Accuracy(binary_mask: Tensor, target_mask: Tensor) -> float:
    #--------------------------------------------------------------------------#
    # Description: Measured pixel accuracy between binary mask and target mask #
    # Input type:                                                              #
    #   - Tensor (binary mask)                                                 #
    #   - Tensor (target mask)                                                 #
    # Return type:                                                             #
    #   - float (result accuracy value)                                        #
    #--------------------------------------------------------------------------#

    return float((((binary_mask == target_mask).sum()) / numel(binary_mask)))



def Dice_Score(predict: array, ground_truth: array) -> float:
    #-------------------------------------------------------------------------#
    # Description: Measured dice score between predict and ground-truth array #
    # Input type:                                                             #
    #   - array (predict output)                                              #
    #   - array (target ground truth)                                         #
    # Return type:                                                            #
    #   - float (result score value)                                          #
    #-------------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Calculate loss value #####
    intersection = sum(predict * ground_truth)
 
    return (((2. * intersection) + 1.) / (sum(predict) + sum(ground_truth) + 1.))



def AUROC_Score(predict: array, ground_truth: array) -> float:
    #--------------------------------------------------------------------------#
    # Description: Measured AUROC score between predict and ground-truth array #
    # Input type:                                                              #
    #   - array (predict output)                                               #
    #   - array (target ground truth)                                          #
    # Return type:                                                             #
    #   - float (result score value)                                           #
    #--------------------------------------------------------------------------#

    return roc_auc_score(y_true = ground_truth, y_score = predict)



def F1_Score(precision: float, recall: float) -> float:
    #--------------------------------------------------------------------#
    # Description: Measured F1 score between precision and recall values #
    # Input type:                                                        #
    #   - float (precision value)                                        #
    #   - float (recall value)                                           #
    # Return type:                                                       #
    #   - float (result score value)                                     #
    #--------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Check if the precision/recall values is valid or not #####
    if ((not precision) and (not recall)):
        return 0.

    return ((2 * precision * recall) / (precision + recall))



def Bbox_IOU(predict: Tensor, ground_truth: Tensor) -> float:
    #----------------------------------------------------------------------------------------------------------#
    # Description: Measured intersection over union (i.e. iou) between predict and ground-truth bounding-boxes #
    # Input type:                                                                                              #
    #   - Tensor (predict bounding-box)                                                                        #
    #   - Tensor (target ground truth bounding-box)                                                            #
    # Return type:                                                                                             #
    #   - float (result intersection over union values)                                                        #
    #----------------------------------------------------------------------------------------------------------#

    ####################
    #Whole process/flow
    ##### Step 1: Calculate intersection over union values #####
    pred_x = max(predict[0], ground_truth[0]) #Predict x-coordinate
    pred_y = max(predict[1], ground_truth[1]) #Predict y-coordinate
    gt_x = min(predict[2], ground_truth[2]) #Ground truth x-coordinate
    gt_y = min(predict[3], ground_truth[3]) #Ground truth y-coordinate

    interArea = (max(0, (gt_x - pred_x)) * max(0, (gt_y - pred_y))) #Compute the area of intersection rectangle

    pred_boxArea = ((predict[2] - predict[0]) * (predict[3] - predict[1])) #Predict area of bounding-boxes
    gt_boxArea = ((ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])) #Ground truth area of bounding-boxes

    return (interArea / float((pred_boxArea + gt_boxArea) - interArea))
    


def AUROC_Curve(predict: array, ground_truth: array) -> tuple:
    #--------------------------------------------------------------------------#
    # Description: Measured AUROC curve between predict and ground-truth array #
    # Input type:                                                              #
    #   - array (predict output)                                               #
    #   - array (target ground truth)                                          #
    # Return type:                                                             #
    #   - tuple (result (false positive rate, true positive rate, _))          #
    #--------------------------------------------------------------------------#

    return roc_curve(y_true = ground_truth, y_score = predict)



def Precision_Recall(confusion_matrix: array) -> Any:
    #------------------------------------------------------------------------------#
    # Description: Measured precision, recall values from related confusion matrix #
    # Input type:                                                                  #
    #   - array (confusion_matrix)                                                 #
    # Return type:                                                                 #
    #   - Any (result precision, recall values)                                    #
    #------------------------------------------------------------------------------#

    ############
    #Initialize
    ##### Length of rows, columns #####
    len_row_column = len(confusion_matrix)

    ##### Result array (i.e. precisions, recalls) #####
    res_prec_arry, res_recl_arry = [], []


    ####################
    #Whole process/flow
    ##### Step 1: Calculate precision, recall values #####
    for row_column_idx in range(1, len_row_column):
        curr_true_post = (confusion_matrix[row_column_idx])[row_column_idx] #Current true-positive indexed-value

        ##### Check if the current indexed-values is zero or not #####
        if (not curr_true_post):
            res_prec_arry.append(0)
            res_recl_arry.append(0)

        else:
            res_prec_arry.append((curr_true_post / (curr_true_post + (sum(confusion_matrix[0: row_column_idx, row_column_idx]) + \
                                  sum(confusion_matrix[(row_column_idx + 1): , row_column_idx])))))
            
            res_recl_arry.append((curr_true_post / (curr_true_post + (sum(confusion_matrix[row_column_idx, 0: row_column_idx]) + \
                                  sum(confusion_matrix[row_column_idx, (row_column_idx + 1): ])))))
        
    return sum(res_prec_arry), sum(res_recl_arry)



def Confusion_Matrix(predict: array, ground_truth: array) -> array:
    #-------------------------------------------------------------------------------#
    # Description: Measured confusion matrix between predict and ground-truth array #
    # Input type:                                                                   #
    #   - array (predict output)                                                    #
    #   - array (target ground truth)                                               #
    # Return type:                                                                  #
    #   - array (result confusion matrix)                                           #
    #-------------------------------------------------------------------------------#

    return confusion_matrix(y_true = ground_truth, y_pred = predict)



def Over_UnderKill_Matrix(predict: List[dict], ground_truth: List[dict], iou_threshold: float, \
                          bbox_iou: float, num_classes: int) -> Any:
    #--------------------------------------------------------------------------------------------#
    # Description: Measured over, under-kill confusion matrix between predicts and ground-truths #
    # Input type:                                                                                #
    #   - List[dict] (predict tensors)                                                           #
    #   - List[dict] (target ground truth tensors)                                               #
    #   - float (self-defined iou-threshold)                                                     #
    #   - float (bounding-boxes iou values)                                                      #
    #   - int (number of labeled classes)                                                        #
    # Return type:                                                                               #
    #   - Any (result confusion matrix informations)                                             #
    #--------------------------------------------------------------------------------------------#

    ############
    #Initialize
    ##### Length of predicts, ground-truth tensors/samples #####
    len_samples = len(ground_truth)

    ##### Record traversal dictionary/hashmap #####
    record_trav_dict = {}

    ##### Record maximun iou values #####
    record_max_iou = 0.

    ##### Result confusion matrix (i.e. over/under-kill, classes) #####
    res_oukill_mat = [[0, 0], [0, 0]]
    res_conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]


    ####################
    #Whole process/flow
    ##### Step 1: Calculate over/under-kill, classes samples quantities distributions #####
    for samples_idx in range(len_samples):
        pred_bboxes = (predict[samples_idx])['boxes'] #Predict bounding-boxes
        gt_bboxes = (ground_truth[samples_idx])['boxes'] #Ground-truth bounding-boxes

        pred_classes = (predict[samples_idx])['labels'] #Predict labels/classes
        gt_classes = (ground_truth[samples_idx])['labels'] #Ground-truth labels/classes

        pred_scores = (predict[samples_idx])['scores'] #Predict labels/classes

        len_pred_bboxes, len_gt_bboxes = len(pred_bboxes), len(gt_bboxes) #Length of predict, ground-truth bounding-boxes
        len_gt_classes = len(gt_classes) #Length of ground-truth labels/classes

        (res_oukill_mat[1])[0] += len_gt_bboxes #Under-kill quantities accumulate

        for gt_class_idx in range(len_gt_classes):
            (res_conf_mat[(gt_classes[gt_class_idx] - 1)])[(num_classes - 1)] += 1 #Under-kill quantities accumulate

        for pred_bbox_idx in range(len_pred_bboxes):

            ##### Non-maximun suppression (i.e. NMS) operations #####
            if (pred_scores[pred_bbox_idx] > iou_threshold):
                for gt_bbox_idx in range(len_gt_bboxes):
                    curr_iou = Bbox_IOU(predict = pred_bboxes[pred_bbox_idx], \
                                        ground_truth = gt_bboxes[gt_bbox_idx]) #Current iou score values
                    
                    ##### Check if the current score values is larger or not #####
                    if (curr_iou > record_max_iou):
                        record_max_iou = curr_iou #Keep updating/overwriting

                        record_idx = gt_bbox_idx #Keep updating/overwriting
                    else:
                        pass
                
                ##### Check if the current iou values matched conditions or not #####
                if (record_max_iou):

                    ##### Check if the current indexed-class matched conditions or not #####
                    if (pred_classes[pred_bbox_idx] == gt_classes[record_idx]):

                        ##### Check if the current indexes existed or not #####
                        if (record_idx not in record_trav_dict):
                            record_trav_dict[record_idx] = record_max_iou #Keep updating/recording

                        else:
                            record_trav_dict[record_idx] += record_max_iou #Keep updating/accumulating

                            ##### Check if the current iou summary exceed boundary or not #####
                            if (record_trav_dict[record_idx] > 1.):
                                (res_oukill_mat[0])[1] += 1 #Over-kill quantities accumulate
                                (res_conf_mat[(num_classes - 1)])[(pred_classes[pred_bbox_idx] - 1)] += 1 #Over-kill quantities accumulate

                            else:
                                pass

                    else:
                        (res_oukill_mat[0])[1] += 1 #Over-kill quantities accumulate
                        (res_conf_mat[(gt_classes[record_idx] - 1)])[(pred_classes[pred_bbox_idx] - 1)] += 1 #Over-kill quantities accumulate

                    record_max_iou = 0. #Reset

                else:
                    (res_oukill_mat[0])[1] += 1 #Over-kill quantities accumulate
                    (res_conf_mat[(num_classes - 1)])[(pred_classes[pred_bbox_idx] - 1)] += 1 #Over-kill quantities accumulate

            else:
                pass

        for sub_record_idx in record_trav_dict:

            ##### Check if the current iou summary is larger than defined-values or not #####
            if (record_trav_dict[sub_record_idx] >= bbox_iou):
                (res_oukill_mat[1])[0] -= 1 #Under-kill quantities reduce
                (res_conf_mat[(gt_classes[sub_record_idx] - 1)])[(num_classes - 1)] -= 1 #Under-kill quantities reduce

                (res_oukill_mat[0])[0] += 1 #Keep updating/accumulating
                (res_oukill_mat[1])[1] += 1 #Keep updating/accumulating
                (res_conf_mat[(gt_classes[sub_record_idx] - 1)])[(gt_classes[sub_record_idx] - 1)] += 1 #Keep updating/accumulating
                (res_conf_mat[(num_classes - 1)])[(num_classes - 1)] += 1 #Keep updating/accumulating
            else:
                (res_oukill_mat[0])[1] += 1 #Over-kill quantities accumulate
                (res_conf_mat[(num_classes - 1)])[(gt_classes[sub_record_idx] - 1)] += 1 #Over-kill quantities accumulate

        ##### Check if the current dictionary/hashmap needs to be reset or not #####
        if (record_trav_dict):
            record_trav_dict.clear() #Reset
        else:
            pass

    return array(res_oukill_mat), array(res_conf_mat)


