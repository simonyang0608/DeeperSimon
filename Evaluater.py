#******************************************************************#
# Source: Evaluater.py                                             #
#                                                                  #
# Description: Major/Main evaluater for model inference/evaluation #
#                                                                  #
# Author: SimonYang                                                #
#******************************************************************#

#================#
# Import Section #
#================#
##########################################
#Pytorch dataloader, transforms functions
from Data.Dataset_Mapper import DataLoader, ToTensor

########################################
#Pytorch nn module (i.e. basic inherit)
from Utility.Ploter import Module

##################################
#Pytorch device, cuda synchronize
from torch import device
from torch.cuda import synchronize

#################
#Pytorch no grad
from torch import no_grad

####################################
#Typing format list, rest functions
from Utility.Metric import (array, Any)

##############################
#Post-process, rest functions
from Utility.PostProcessor import (Mask_To_Bbox_Multi, Mask_To_Bbox_Single, \
                                   Draw_Bbox_Det, Draw_Bbox_Seg_Multi, \
                                   Draw_Bbox_Seg_Single, Filter_Target, \
                                   Filter_Outputs, cvtColor)

####################################
#File type/format (i.e. video, ...)
from filetype import is_video

###################
#Tqdm progress bar
from tqdm import tqdm

############################
#Operating system (i.e. OS)
from os.path import join

################
#Measured timer
from time import time


#=====================#
# Class Function List #
#=====================#
class Major_Evaluater(object):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, num_classes: int, camr_scl: float, ref_scl: list, \
                 post_scl: list) -> None:
        #-----------------------------------------------#
        # Description: Constructor initialize/setup     #
        # Input type:                                   # 
        #   - int (number of labeled classes)           #
        #   - float (filtered-scale (i.e. camera))      #
        #   - list (filtered-scale (i.e. reference))    #
        #   - list (filtered-scale (i.e. post-process)) #
        # Return type:                                  #
        #   - None (void, no return)                    #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### Metric array (i.e. un-filtered/filtered) #####
        self.gt_arry = []
        self.filt_gt_arry = []

        self.gt_bbox_arry = []
        self.filt_gt_bbox_arry = []

        self.pred_arry = []
        self.filt_pred_arry = []

        self.pred_bbox_arry = []
        self.filt_pred_bbox_arry = []

        ##### Elapsed-time array #####
        self.elapsed_time_arry = []

        ##### Color array #####
        self.color_arry = [(0, 0, 255), (0, 255, 0), \
                           (128, 0, 255), (130, 238, 238), \
                           (128, 0, 0), (0, 128, 128), \
                           (117, 89, 175), (112, 112, 255), \
                           (0, 192, 192), (0, 101, 255), \
                           (128, 0, 128), (128, 128, 0), \
                           (70, 0, 79), (175, 90, 220), \
                           (144, 255, 58), (64, 64, 64), \
                           (255, 0, 0)]

        ##### Transforms #####
        self.transform = ToTensor()

        ##### Number of classes #####
        self.num_classes = num_classes

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        self.norm_ref_scl1_w, self.norm_ref_scl1_h = (ref_scl[0][0] / camr_scl), \
                                                     (ref_scl[0][1] / camr_scl)
        self.norm_ref_scl2_w, self.norm_ref_scl2_h = (ref_scl[1][0] / camr_scl), \
                                                     (ref_scl[1][1] / camr_scl)
        self.norm_ref_scl3_w, self.norm_ref_scl3_h = (ref_scl[2][0] / camr_scl), \
                                                     (ref_scl[2][1] / camr_scl)

        self.norm_post_scl1_w, self.norm_post_scl1_h = (post_scl[0][0] / camr_scl), \
                                                       (post_scl[0][1] / camr_scl)
        self.norm_post_scl2_w, self.norm_post_scl2_h = (post_scl[1][0] / camr_scl), \
                                                       (post_scl[1][1] / camr_scl)
        self.norm_post_scl3_w, self.norm_post_scl3_h = (post_scl[2][0] / camr_scl), \
                                                       (post_scl[2][1] / camr_scl)

        ##### Filtered-outputs/target (i.e. faster r-cnn detection) #####
        self.filt_outputs = []
        self.filt_target = []

        ##### Record maximun/minimun losses, info. (i.e. FCOS, MemSeg masks) #####
        self.record_max_loss = -2.0
        self.record_min_loss = 2.0

        self.record_max_info = [-2.0]
        self.record_min_info = [2.0]


    ########################
    # Member Function List #
    ########################
    def dataloader_evaluater(self, eval_loader: DataLoader, model: Module, criterion: Module, \
                             device: device, logger: Any, task: str, fpath: str) -> None:
        #------------------------------------------------------#
        # Description: Dataset loader full file path evaluater #
        # Input type:                                          #
        #   - DataLoader (evaluate dataset loader)             #
        #   - Module (self-defined model)                      #
        #   - Module (self-defined criterion/loss function)    #
        #   - device (gpu/cpu device)                          #
        #   - Any (logging record)                             #
        #   - str (tasks type)                                 #
        #   - str (file path)                                  #
        # Return type:                                         #
        #   - None (void, no return)                           #
        #------------------------------------------------------#
        from Utility.Metric import (Binary_Pixel_Accuracy, AUROC_Score, Dice_Score, \
                                    F1_Score, AUROC_Curve, Confusion_Matrix, \
                                    Over_UnderKill_Matrix, Precision_Recall) #Temporal import measured-metrics
        
        from Utility.Ploter import (SummaryWriter, Plot_AUROC_Curve, Plot_Confusion_Matrix, \
                                    Add_Scalar, Add_Scalars, Add_Image, Add_Figure) #Temporal import tensorboard, data-visualizations

        ############
        #Initialize
        ##### Switch model to validate mode #####
        model.eval()

        ##### Evaluate with tqdm progress-bar #####
        t_epoch = tqdm(eval_loader)

        t_epoch.set_description("Epoch [{}/{}] -> Evaluate" \
                                .format(0, 0)) #Set tqdm description

        ##### Metric array (i.e. un-filtered/filtered) #####
        gt_arry = self.gt_arry
        filt_gt_arry = self.filt_gt_arry

        gt_bbox_arry = self.gt_bbox_arry
        filt_gt_bbox_arry = self.filt_gt_bbox_arry

        pred_arry = self.pred_arry
        filt_pred_arry = self.filt_pred_arry

        pred_bbox_arry = self.pred_bbox_arry
        filt_pred_bbox_arry = self.filt_pred_bbox_arry

        ##### Number of classes #####
        num_classes = self.num_classes

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        norm_ref_scl = [[self.norm_ref_scl1_w, self.norm_ref_scl1_h], \
                        [self.norm_ref_scl2_w, self.norm_ref_scl2_h], \
                        [self.norm_ref_scl3_w, self.norm_ref_scl3_h]]
        
        norm_post_scl = [[self.norm_post_scl1_w, self.norm_post_scl1_h], \
                         [self.norm_post_scl2_w, self.norm_post_scl2_h], \
                         [self.norm_post_scl3_w, self.norm_post_scl3_h]]

        ##### Filtered-outputs/target (i.e. faster r-cnn detection) #####
        filt_outputs = self.filt_outputs
        filt_target = self.filt_target

        ##### Record maximun/minimun losses, info. (i.e. FCOS, MemSeg masks) #####
        record_max_loss = self.record_max_loss
        record_min_loss = self.record_min_loss

        record_max_info = self.record_max_info
        record_min_info = self.record_min_info

        ##### Visualized writer #####
        writer = SummaryWriter(fpath)


        #################################################
        #Customized evaluate status with different tasks
        ##### Detection #####
        if (task == 'detection'):
            for (image, target) in t_epoch:
                image = list(sub_image.to(device) for \
                             sub_image in image) #Re-create aligned batch images list

                target = list({sub_key: sub_value.to(device) for \
                               sub_key, sub_value in sub_target.items()} for \
                               sub_target in target) #Re-create aligned batch targets list
                
                with no_grad():
                    outputs = model(image, None) #Predict labels, bboxes, scores output

                Filter_Target(norm_ref_scl = norm_ref_scl, target = target, filt_target = filt_target) #Filtered-out target list
                Filter_Outputs(norm_post_scl = norm_post_scl, outputs = outputs, filt_outputs = filt_outputs) #Filtered-out outputs list

                gt_arry.extend(target) #Predict tensors extensions
                pred_arry.extend(outputs) #Ground-truth tensors extensions

                filt_gt_arry.extend(filt_target) #Filtered-predict tensors extensions
                filt_pred_arry.extend(filt_outputs) #Filtered ground-truth tensors extensions

                model.train() #Switch model to train mode

                outputs = model(image, target) #Loss output
                
                loss_value = sum(sub_loss for sub_loss in outputs.values()) #Loss value calculate

                rpn_cls_loss, rpn_bbox_loss, total_loss = \
                                    float(outputs['loss_objectness']), \
                                    float(outputs['loss_rpn_box_reg']), \
                                    float(loss_value)
                
                model.eval() #Switch model to validate mode

                if (filt_target):
                    filt_target.clear() #Reset
                else:
                    pass

                if (filt_outputs):
                    filt_outputs.clear() #Reset
                else:
                    pass

                t_epoch.set_postfix(rpn_cls_loss = rpn_cls_loss, rpn_bbox_loss = rpn_bbox_loss, \
                                    total_loss = total_loss)
                

            ##### Visualized metrics from output results #####
            over_underkill_matrix_50, confusion_matrix_50 = \
                Over_UnderKill_Matrix(predict = pred_arry, ground_truth = gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.5, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.5)
            filt_over_underkill_matrix_50, filt_confusion_matrix_50 = \
                Over_UnderKill_Matrix(predict = filt_pred_arry, ground_truth = filt_gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.5, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.5)
                    
            over_underkill_matrix_75, confusion_matrix_75 = \
                Over_UnderKill_Matrix(predict = pred_arry, ground_truth = gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.75, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.75)
            filt_over_underkill_matrix_75, filt_confusion_matrix_75 = \
                Over_UnderKill_Matrix(predict = filt_pred_arry, ground_truth = filt_gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.75, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.75)
                    
            over_underkill_matrix_85, confusion_matrix_85 = \
                Over_UnderKill_Matrix(predict = pred_arry, ground_truth = gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.85, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.85)
            filt_over_underkill_matrix_85, filt_confusion_matrix_85 = \
                Over_UnderKill_Matrix(predict = filt_pred_arry, ground_truth = filt_gt_arry, iou_threshold = 0.28, \
                                      bbox_iou = 0.85, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.85)
                        
            precision_50, recall_50 = Precision_Recall(confusion_matrix = over_underkill_matrix_50) #Precision, Recall values (i.e. iou threshold = 0.5)
            filt_precision_50, filt_recall_50 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_50) #Filtered-precision, Recall values (i.e. iou threshold = 0.5)

            precision_75, recall_75 = Precision_Recall(confusion_matrix = over_underkill_matrix_75) #Precision, Recall values (i.e. iou threshold = 0.75)
            filt_precision_75, filt_recall_75 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_75) #Filtered-precision, Recall values (i.e. iou threshold = 0.75)

            precision_85, recall_85 = Precision_Recall(confusion_matrix = over_underkill_matrix_85) #Precision, Recall values (i.e. iou threshold = 0.85)
            filt_precision_85, filt_recall_85 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_85) #Filtered-precision, Recall values (i.e. iou threshold = 0.85)

            f1_score_50 = F1_Score(precision = precision_50, recall = recall_50) #F1 score (i.e. iou threshold = 0.5)
            filt_f1_score_50 = F1_Score(precision = filt_precision_50, recall = filt_recall_50) #Filtered-f1 score (i.e. iou threshold = 0.5)

            f1_score_75 = F1_Score(precision = precision_75, recall = recall_75) #F1 score (i.e. iou threshold = 0.75)
            filt_f1_score_75 = F1_Score(precision = filt_precision_75, recall = filt_recall_75) #Filtered-f1 score (i.e. iou threshold = 0.75)

            f1_score_85 = F1_Score(precision = precision_85, recall = recall_85) #F1 score (i.e. iou threshold = 0.85)
            filt_f1_score_85 = F1_Score(precision = filt_precision_85, recall = filt_recall_85) #Filtered-f1 score (i.e. iou threshold = 0.85)

            Add_Scalars(writer = writer, title = 'Precision/50', dict_value = {'Un-Filtered': precision_50, \
                        'Filtered': filt_precision_50}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
            
            Add_Scalars(writer = writer, title = 'Precision/75', dict_value = {'Un-Filtered': precision_75, \
                        'Filtered': filt_precision_75}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)

            Add_Scalars(writer = writer, title = 'Precision/85', dict_value = {'Un-Filtered': precision_85, \
                        'Filtered': filt_precision_85}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

            Add_Scalars(writer = writer, title = 'Recall/50', dict_value = {'Un-Filtered': recall_50, \
                        'Filtered': filt_recall_50}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
            
            Add_Scalars(writer = writer, title = 'Recall/75', dict_value = {'Un-Filtered': recall_75, \
                        'Filtered': filt_recall_75}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
            
            Add_Scalars(writer = writer, title = 'Recall/85', dict_value = {'Un-Filtered': recall_85, \
                        'Filtered': filt_recall_85}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

            Add_Scalars(writer = writer, title = 'F1_Score/50', dict_value = {'Un-Filtered': f1_score_50, \
                        'Filtered': filt_f1_score_50}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
            
            Add_Scalars(writer = writer, title = 'F1_Score/75', dict_value = {'Un-Filtered': f1_score_75, \
                        'Filtered': filt_f1_score_75}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
            
            Add_Scalars(writer = writer, title = 'F1_Score/85', dict_value = {'Un-Filtered': f1_score_85, \
                        'Filtered': filt_f1_score_85}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)
        
            Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       over_underkill_matrix_50), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
            Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_over_underkill_matrix_50), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
            
            Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       over_underkill_matrix_75), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
            Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_over_underkill_matrix_75), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
            
            Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       over_underkill_matrix_85), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
            Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_over_underkill_matrix_85), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
            
            Add_Figure(writer = writer, title = 'Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       confusion_matrix_50), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.5)
            Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_confusion_matrix_50), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.5)
            
            Add_Figure(writer = writer, title = 'Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       confusion_matrix_75), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.75)
            Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_confusion_matrix_75), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.75)
            
            Add_Figure(writer = writer, title = 'Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       confusion_matrix_85), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.85)
            Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                       filt_confusion_matrix_85), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.85)

            logger.info('Epoch [{}/{}] -> Validate: rpn_cls_loss: {}  rpn_bbox_loss: {}  total_loss: {}' \
                        .format(0, 0, rpn_cls_loss, rpn_bbox_loss, total_loss))
            
            print('===================================================')


        ##### Segmentation #####
        elif (task == 'segmentation'):
            for (image, mask) in t_epoch:
                image = image.to(device) #Point image data to main device
                mask = mask.to(device) #Point mask data to main device

                batch_range = (image.shape)[0] #Batch range sizes

                if (num_classes > 1): #Multiple classes
                    with no_grad():
                        pred_mask, pred_argmax = model(image) #Predict mask output

                    loss_value = (criterion(pred_mask, mask)).float() #Loss value calculate

                    argmax_mask = (((pred_argmax.squeeze(1)).detach()).cpu()).numpy() #Argument-max mask output
                    target_mask = ((mask.detach()).cpu()).numpy() #Target mask output

                    for batch_idx in range(batch_range):
                        gt_dict = {'boxes': [], 'labels': [], 'scores': []} #Ground-truth dictionary/hashmap
                        filt_gt_dict = {'boxes': [], 'labels': [], 'scores': []} #Filtered-ground-truth dictionary/hashmap

                        pred_dict = {'boxes': [], 'labels': [], 'scores': []} #Predict dictionary/hashmap
                        filt_pred_dict = {'boxes': [], 'labels': [], 'scores': []} #Filtered-predict dictionary/hashmap

                        sub_argmax_mask, sub_target_mask = argmax_mask[batch_idx], target_mask[batch_idx] #Sub argument-max, target mask

                        Mask_To_Bbox_Multi(gt_dict = gt_dict, filt_gt_dict = filt_gt_dict, pred_dict = pred_dict, \
                                           filt_pred_dict = filt_pred_dict, sub_argmax_mask = sub_argmax_mask, \
                                           sub_target_mask = sub_target_mask, norm_ref_scl = norm_ref_scl, \
                                           norm_post_scl = norm_post_scl, num_classes = num_classes) #Mask to bbox conversions (i.e. un-filtered/filtered)

                        gt_bbox_arry.append(gt_dict)
                        filt_gt_bbox_arry.append(filt_gt_dict)

                        pred_bbox_arry.append(pred_dict)
                        filt_pred_bbox_arry.append(filt_pred_dict)

                    if (loss_value > record_max_loss): #Track for maximun loss
                        record_max_loss = loss_value

                        record_max_info.clear() #Reset

                        record_max_info.append(image)
                        record_max_info.append(((mask.unsqueeze(1)) / (num_classes - 1)))
                        record_max_info.append(((pred_argmax.unsqueeze(1)) / (num_classes - 1)))
                    else:
                        pass

                    if (loss_value < record_min_loss): #Track for minimun loss
                        record_min_loss = loss_value

                        record_min_info.clear() #Reset

                        record_min_info.append(image)
                        record_min_info.append(((mask.unsqueeze(1)) / (num_classes - 1)))
                        record_min_info.append(((pred_argmax.unsqueeze(1)) / (num_classes - 1)))
                    else:
                        pass

                    pred_arry.extend((pred_argmax.view(-1)).tolist())

                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = pred_argmax, target_mask = mask) #Accuracy value calculate

                else: #Single class
                    with no_grad():
                        pred_mask = model(image) #Predict mask output

                    loss_value = (criterion(pred_mask, mask)).float() #Loss value calculate

                    binary_mask = (pred_mask > 0.5).float() #Binary mask output

                    pred_binmask = (((binary_mask.squeeze(1)).detach()).cpu()).numpy() #Predict binary mask output
                    gt_binmask = (((mask.squeeze(1)).detach()).cpu()).numpy() #Ground-truth binary mask output

                    for batch_idx in range(batch_range):
                        gt_dict = {'boxes': [], 'labels': [], 'scores': []} #Ground-truth dictionary/hashmap
                        filt_gt_dict = {'boxes': [], 'labels': [], 'scores': []} #Filtered-ground-truth dictionary/hashmap

                        pred_dict = {'boxes': [], 'labels': [], 'scores': []} #Predict dictionary/hashmap
                        filt_pred_dict = {'boxes': [], 'labels': [], 'scores': []} #Filtered-predict dictionary/hashmap

                        sub_gt_binmask, sub_pred_binmask = gt_binmask[batch_idx], pred_binmask[batch_idx] #Sub ground-truth, predict binary mask

                        Mask_To_Bbox_Single(gt_dict = gt_dict, filt_gt_dict = filt_gt_dict, pred_dict = pred_dict, \
                                            filt_pred_dict = filt_pred_dict, sub_gt_binmask = sub_gt_binmask, \
                                            sub_pred_binmask = sub_pred_binmask, norm_ref_scl = norm_ref_scl, \
                                            norm_post_scl = norm_post_scl) #Mask to bbox conversions (i.e. un-filtered/filtered)

                        gt_bbox_arry.append(gt_dict)
                        filt_gt_bbox_arry.append(filt_gt_dict)

                        pred_bbox_arry.append(pred_dict)
                        filt_pred_bbox_arry.append(filt_pred_dict)

                    if (loss_value > record_max_loss): #Track for maximun loss
                        record_max_loss = loss_value

                        record_max_info.clear() #Reset

                        record_max_info.append(image)
                        record_max_info.append(mask)
                        record_max_info.append(binary_mask)
                    else:
                        pass

                    if (loss_value < record_min_loss): #Track for minimun loss
                        record_min_loss = loss_value

                        record_min_info.clear() #Reset

                        record_min_info.append(image)
                        record_min_info.append(mask)
                        record_min_info.append(binary_mask)
                    else:
                        pass

                    pred_arry.extend((binary_mask.view(-1)).tolist())

                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = binary_mask, target_mask = mask) #Accuracy value calculate

                gt_arry.extend((mask.view(-1)).tolist())

                t_epoch.set_postfix(loss = loss_value.item(), accuracy = (100. * accuracy_value))
            

            ##### Visualized metrics from output results #####
            if (len(set(gt_arry)) != 1):
                pred_np_arry, gt_np_arry = array(pred_arry), array(gt_arry) #Numpy array conversions (i.e. predict, ground-truth)

                if (num_classes == 1): #Single class
                    dice_score_value = Dice_Score(predict = pred_np_arry, ground_truth = gt_np_arry) #Dice score
                    auroc_score_value = AUROC_Score(predict = pred_np_arry, ground_truth = gt_np_arry) #AUROC score
                    fp_rate, tp_rate, _ = AUROC_Curve(predict = pred_np_arry, ground_truth = gt_np_arry) #AUROC curve

                    Add_Scalar(writer = writer, title = 'Dice_Score', value = dice_score_value, \
                               epoch = 0) #Dice score curve visualized

                    Add_Scalar(writer = writer, title = 'AUROC_Score', value = auroc_score_value, \
                               epoch = 0) #AUROC score curve visualized

                    Add_Figure(writer = writer, title = 'AUROC_Curve', figure = Plot_AUROC_Curve(fp_rate = fp_rate, \
                               tp_rate = tp_rate, auroc_score = auroc_score_value), epoch = 0) #AUROC area curve visualized
                    
                else: #Multiple classes
                    pass

                confusion_matrix = Confusion_Matrix(predict = pred_np_arry, ground_truth = gt_np_arry) #Confusion Matrix

                over_underkill_matrix_50, confusion_matrix_50 = \
                    Over_UnderKill_Matrix(predict = pred_bbox_arry, ground_truth = gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.5, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.5)
                filt_over_underkill_matrix_50, filt_confusion_matrix_50 = \
                    Over_UnderKill_Matrix(predict = filt_pred_bbox_arry, ground_truth = filt_gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.5, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.5)
                    
                over_underkill_matrix_75, confusion_matrix_75 = \
                    Over_UnderKill_Matrix(predict = pred_bbox_arry, ground_truth = gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.75, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.75)
                filt_over_underkill_matrix_75, filt_confusion_matrix_75 = \
                    Over_UnderKill_Matrix(predict = filt_pred_bbox_arry, ground_truth = filt_gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.75, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.75)
                        
                over_underkill_matrix_85, confusion_matrix_85 = \
                    Over_UnderKill_Matrix(predict = pred_bbox_arry, ground_truth = gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.85, num_classes = (num_classes + 1)) #Metric matrixs (i.e. iou threshold = 0.85)
                filt_over_underkill_matrix_85, filt_confusion_matrix_85 = \
                    Over_UnderKill_Matrix(predict = filt_pred_bbox_arry, ground_truth = filt_gt_bbox_arry, iou_threshold = 0.28, \
                                          bbox_iou = 0.85, num_classes = (num_classes + 1)) #Filtered-metric matrixs (i.e. iou threshold = 0.85)
                            
                precision_50, recall_50 = Precision_Recall(confusion_matrix = over_underkill_matrix_50) #Precision, Recall values (i.e. iou threshold = 0.5)
                filt_precision_50, filt_recall_50 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_50) #Filtered-precision, Recall values (i.e. iou threshold = 0.5)

                precision_75, recall_75 = Precision_Recall(confusion_matrix = over_underkill_matrix_75) #Precision, Recall values (i.e. iou threshold = 0.75)
                filt_precision_75, filt_recall_75 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_75) #Filtered-precision, Recall values (i.e. iou threshold = 0.75)

                precision_85, recall_85 = Precision_Recall(confusion_matrix = over_underkill_matrix_85) #Precision, Recall values (i.e. iou threshold = 0.85)
                filt_precision_85, filt_recall_85 = Precision_Recall(confusion_matrix = filt_over_underkill_matrix_85) #Filtered-precision, Recall values (i.e. iou threshold = 0.85)

                f1_score_50 = F1_Score(precision = precision_50, recall = recall_50) #F1 score (i.e. iou threshold = 0.5)
                filt_f1_score_50 = F1_Score(precision = filt_precision_50, recall = filt_recall_50) #Filtered-f1 score (i.e. iou threshold = 0.5)

                f1_score_75 = F1_Score(precision = precision_75, recall = recall_75) #F1 score (i.e. iou threshold = 0.75)
                filt_f1_score_75 = F1_Score(precision = filt_precision_75, recall = filt_recall_75) #Filtered-f1 score (i.e. iou threshold = 0.75)

                f1_score_85 = F1_Score(precision = precision_85, recall = recall_85) #F1 score (i.e. iou threshold = 0.85)
                filt_f1_score_85 = F1_Score(precision = filt_precision_85, recall = filt_recall_85) #Filtered-f1 score (i.e. iou threshold = 0.85)

                Add_Scalars(writer = writer, title = 'Precision/50', dict_value = {'Un-Filtered': precision_50, \
                            'Filtered': filt_precision_50}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                
                Add_Scalars(writer = writer, title = 'Precision/75', dict_value = {'Un-Filtered': precision_75, \
                            'Filtered': filt_precision_75}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)

                Add_Scalars(writer = writer, title = 'Precision/85', dict_value = {'Un-Filtered': precision_85, \
                            'Filtered': filt_precision_85}, epoch = 0) #Precision score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                Add_Scalars(writer = writer, title = 'Recall/50', dict_value = {'Un-Filtered': recall_50, \
                            'Filtered': filt_recall_50}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                
                Add_Scalars(writer = writer, title = 'Recall/75', dict_value = {'Un-Filtered': recall_75, \
                            'Filtered': filt_recall_75}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                
                Add_Scalars(writer = writer, title = 'Recall/85', dict_value = {'Un-Filtered': recall_85, \
                            'Filtered': filt_recall_85}, epoch = 0) #Recall score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                Add_Scalars(writer = writer, title = 'F1_Score/50', dict_value = {'Un-Filtered': f1_score_50, \
                            'Filtered': filt_f1_score_50}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                
                Add_Scalars(writer = writer, title = 'F1_Score/75', dict_value = {'Un-Filtered': f1_score_75, \
                            'Filtered': filt_f1_score_75}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                
                Add_Scalars(writer = writer, title = 'F1_Score/85', dict_value = {'Un-Filtered': f1_score_85, \
                            'Filtered': filt_f1_score_85}, epoch = 0) #F1 score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                Add_Figure(writer = writer, title = 'Pixel_based_Confusion_Matrix', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           confusion_matrix), epoch = 0) #Confusion matrix visualized (i.e. pixel-based)
            
                Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           over_underkill_matrix_50), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_over_underkill_matrix_50), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                
                Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           over_underkill_matrix_75), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_over_underkill_matrix_75), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                
                Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           over_underkill_matrix_85), epoch = 0) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_over_underkill_matrix_85), epoch = 0) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                
                Add_Figure(writer = writer, title = 'Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           confusion_matrix_50), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.5)
                Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_confusion_matrix_50), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.5)
                
                Add_Figure(writer = writer, title = 'Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           confusion_matrix_75), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.75)
                Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_confusion_matrix_75), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.75)
                
                Add_Figure(writer = writer, title = 'Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           confusion_matrix_85), epoch = 0) #Confusion matrix visualized (i.e. iou threshold = 0.85)
                Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                           filt_confusion_matrix_85), epoch = 0) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.85)
            
            else:
                pass

            Add_Image(writer = writer, title = 'High_Loss/Images', image = record_max_info[0], \
                      epoch = 0) #High loss batch images visualized
            Add_Image(writer = writer, title = 'Low_Loss/Images', image = record_min_info[0], \
                      epoch = 0) #Low loss batch images visualized
            
            Add_Image(writer = writer, title = 'High_Loss/Target_Masks', image = record_max_info[1], \
                      epoch = 0) #High loss batch target masks visualized
            Add_Image(writer = writer, title = 'Low_Loss/Target_Masks', image = record_min_info[1], \
                      epoch = 0) #Low loss batch target masks visualized
            
            Add_Image(writer = writer, title = 'High_Loss/Predict_Masks', image = record_max_info[2], \
                      epoch = 0) #High loss batch predict masks visualized
            Add_Image(writer = writer, title = 'Low_Loss/Predict_Masks', image = record_min_info[2], \
                      epoch = 0) #Low loss batch predict masks visualized
            
            logger.info('Epoch [{}/{}] -> Evaluate: accuracy: {}  loss: {}'.format(0, 0, (100. * accuracy_value), \
                        loss_value.item()))
            
            print('===================================================')


        ##### Classification #####
        elif (task == 'classification'):
            pass


    def imagefolder_evaluater(self, fpath: str, model: Module, device: device, task: str, \
                              target_fdir: str, logger: Any, defect_dict: dict) -> None:
        #-----------------------------------------------#
        # Description: Image folder full path evaluater #
        # Input type:                                   #
        #   - str (folder path)                         #    
        #   - Module (self-defined model)               #  
        #   - device (gpu/cpu device)                   #
        #   - str (tasks type)                          #
        #   - str (target file/folder directory)        #
        #   - Any (logging record)                      #
        #   - dict (defect defined dictionary)          #
        # Return type:                                  #
        #   - None (void, no return)                    #
        #-----------------------------------------------#
        from PIL.Image import open #Temporal import pillow functions
        from Utility.PostProcessor import imwrite, COLOR_RGB2BGR #Temporal import image, rest functions
        from filetype import is_image #Temporal import file type/format (i.e. image, ...)
        from os import listdir #Temporal import operating system functions

        ############
        #Initialize
        ##### Switch model to validate mode #####
        model.eval()

        ##### Directory file path #####
        src_fpath = listdir(fpath)

        ##### Evaluate with tqdm progress-bar #####
        t_pbar = tqdm(range(len(src_fpath)))

        t_pbar.set_description("Evaluate") #Set tqdm description

        ##### Elapsed-time array #####
        elapsed_time_arry = self.elapsed_time_arry

        ##### Color array #####
        color_arry = self.color_arry

        ##### Transforms #####
        transform = self.transform

        ##### Number of classes #####
        num_classes = self.num_classes

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        norm_post_scl = [[self.norm_post_scl1_w, self.norm_post_scl1_h], \
                         [self.norm_post_scl2_w, self.norm_post_scl2_h], \
                         [self.norm_post_scl3_w, self.norm_post_scl3_h]]

        ##### Defect classes list #####
        defect_list = list(defect_dict.keys())


        #################################################
        #Customized evaluate status with different tasks
        for idx in t_pbar:
            img_file = join(fpath, src_fpath[idx]) #Image file name

            ##### Check if the current file is valid image-type or not #####
            if (is_image(img_file)):
                img_rgb = open(img_file) #RGB image (pillow)

                if (img_rgb.mode != 'RGB'): #Check if input image is in RGB format or not
                    img_rgb = img_rgb.convert('RGB')

                else:
                    pass

                img_bgr = cvtColor(array(img_rgb), COLOR_RGB2BGR) #Convert RGB to BGR image (opencv)

                img = transform(img_rgb) #Transforms normalize
                img = img.unsqueeze(0) #Generate one-batch tensor shape

                img = img.to(device) #Point image data to main device

                start_time = time() #Start time
                synchronize() #Timer synchronize


                ##### Detection #####
                if (task == 'detection'):
                    with no_grad():
                        outputs = model(img, None) #Predict labels, bboxes, scores output

                    elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                    len_outputs = len(outputs) #Length of predict outputs

                    for tmp_idx in range(len_outputs):
                        pred_scores = (outputs[tmp_idx])['scores'] #Predict scores
                        pred_boxes = (outputs[tmp_idx])['boxes'] #Predict bboxes
                        pred_labels = (outputs[tmp_idx])['labels'] #Predict labels

                        len_pred_scores = len(pred_scores) #Length of predict scores

                        Draw_Bbox_Det(len_pred_scores = len_pred_scores, iou_threshold = 0.2, \
                                      img_bgr = img_bgr, defect_list = defect_list, \
                                      color_arry = color_arry, norm_post_scl = norm_post_scl, \
                                      pred_scores = pred_scores, pred_boxes = pred_boxes, \
                                      pred_labels = pred_labels, show_filtered = True) #Draw-on bounding-boxes


                ##### Segmentation #####
                elif (task == 'segmentation'):
                    if (num_classes > 1): #Multiple classes
                        with no_grad():
                            _, pred_argmax = model(img) #Predict mask output

                        elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                        argmax_mask = (((pred_argmax.squeeze()).detach()).cpu()).numpy() #Argument-max mask output
                        
                        Draw_Bbox_Seg_Multi(argmax_mask = argmax_mask, img_bgr = img_bgr, \
                                            defect_list = defect_list, color_arry = color_arry, \
                                            norm_post_scl = norm_post_scl, num_classes = num_classes, \
                                            show_filtered = True) #Draw-on bounding-boxes

                    else: #Single class
                        with no_grad():
                            pred_mask = model(img) #Predict mask output

                        elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                        binary_mask = ((((pred_mask.squeeze() > 0.5).detach()).cpu()).numpy()).astype('uint8') #Binary mask output

                        Draw_Bbox_Seg_Single(binary_mask = binary_mask, img_bgr = img_bgr, \
                                             defect_list = defect_list, color_arry = color_arry, \
                                             norm_post_scl = norm_post_scl, show_filtered = True) #Draw-on bounding-boxes


                ##### Classification #####
                elif (task == 'classification'):
                    pass
                
                imwrite(join(target_fdir, src_fpath[idx]), img_bgr) #Save to target directory/folder

            else:
                pass

        print('===================================================')


        logger.info('=> Done! Output total of {} results to target folder directory completed!!'.format(len(src_fpath)))

        logger.info('==> Total elapsed time: {} seconds'.format(sum(elapsed_time_arry)))
        logger.info('==> Average inference speed: {} seconds/image'.format((sum(elapsed_time_arry) / \
                                                                            len(elapsed_time_arry))))
        

    def videowriter_evaluater(self, fpath: str, model: Module, device: device, task: str, \
                              target_fdir: str, logger: Any, defect_dict: dict) -> None:
        #-----------------------------------------------#
        # Description: Video writer full path evaluater #
        # Input type:                                   #
        #   - str (file path)                           #    
        #   - Module (self-defined model)               #  
        #   - device (gpu/cpu device)                   #
        #   - str (tasks type)                          #
        #   - str (target file/folder directory)        #
        #   - Any (logging record)                      #
        #   - dict (defect defined dictionary)          #
        # Return type:                                  #
        #   - None (void, no return)                    #
        #-----------------------------------------------#
        from PIL.Image import fromarray #Temporal import pillow functions
        from cv2 import (VideoCapture, VideoWriter, VideoWriter_fourcc) #Temporal import opencv (i.e. cv2) functions
        from Utility.PostProcessor import COLOR_BGR2RGB #Temporal import image, rest functions
        
        ############
        #Initialize
        ##### ##### Switch model to validate mode #####
        model.eval()

        ##### Video captured-input #####
        vid_capin = VideoCapture(fpath)

        ##### Evaluate with tqdm progress-bar #####
        t_pbar = tqdm(range(int(vid_capin.get(7))))

        t_pbar.set_description("Evaluate") #Set tqdm description

        ##### Elapsed-time array #####
        elapsed_time_arry = self.elapsed_time_arry

        ##### Color array #####
        color_arry = self.color_arry

        ##### Transforms #####
        transform = self.transform

        ##### Number of classes #####
        num_classes = self.num_classes

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        norm_post_scl = [[self.norm_post_scl1_w, self.norm_post_scl1_h], \
                         [self.norm_post_scl2_w, self.norm_post_scl2_h], \
                         [self.norm_post_scl3_w, self.norm_post_scl3_h]]

        ##### Output video-writer format, result #####
        vid_fname = (((fpath.split('/'))[(-1)]).split('.'))[0] #Video file name

        outvid_writer = VideoWriter((join(target_fdir, vid_fname) + ".avi"), VideoWriter_fourcc(*'XVID'), \
                                     int(vid_capin.get(5)), (int(vid_capin.get(3)), int(vid_capin.get(4)))) #Output video-writer in MP4 format
        
        ##### Defect classes list #####
        defect_list = list(defect_dict.keys())
        

        #################################################
        #Customized evaluate status with different tasks
        for _ in t_pbar:
            ret, frame_bgr = vid_capin.read() #Video frame captured

            ##### Check if the current video-straming is valid or not #####
            if (not ret):
                logger.error("Invalid image type/format for streaming ! Exit ...")

                break

            frame_rgb = cvtColor(array(frame_bgr), COLOR_BGR2RGB) #Convert BGR to RGB image (opencv)
            frame_rgb = fromarray(frame_rgb)

            frame_rgb = transform(frame_rgb) #Transforms normalize
            frame_rgb = frame_rgb.unsqueeze(0) #Generate one-batch tensor shape

            frame_rgb = frame_rgb.to(device) #Point frame data to main device

            start_time = time() #Start time
            synchronize() #Timer synchronize


            ##### Detection #####
            if (task == 'detection'):
                with no_grad():
                    outputs = model(frame_rgb, None) #Predict labels, bboxes, scores output

                elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                len_outputs = len(outputs) #Length of predict outputs

                for tmp_idx in range(len_outputs):
                    pred_scores = (outputs[tmp_idx])['scores'] #Predict scores
                    pred_boxes = (outputs[tmp_idx])['boxes'] #Predict bboxes
                    pred_labels = (outputs[tmp_idx])['labels'] #Predict labels

                    len_pred_scores = len(pred_scores) #Length of predict scores

                    Draw_Bbox_Det(len_pred_scores = len_pred_scores, iou_threshold = 0.2, \
                                  img_bgr = frame_bgr, defect_list = defect_list, \
                                  color_arry = color_arry, norm_post_scl = norm_post_scl, \
                                  pred_scores = pred_scores, pred_boxes = pred_boxes, \
                                  pred_labels = pred_labels, show_filtered = False) #Draw-on bounding-boxes


            ##### Segmentation #####
            elif (task == 'segmentation'):
                if (num_classes > 1): #Multiple classes
                    with no_grad():
                        _, pred_argmax = model(frame_rgb) #Predict mask output

                    elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                    argmax_mask = (((pred_argmax.squeeze()).detach()).cpu()).numpy() #Argument-max mask output
                    
                    Draw_Bbox_Seg_Multi(argmax_mask = argmax_mask, img_bgr = frame_bgr, \
                                        defect_list = defect_list, color_arry = color_arry, \
                                        norm_post_scl = norm_post_scl, num_classes = num_classes, \
                                        show_filtered = False) #Draw-on bounding-boxes

                else: #Single class
                    with no_grad():
                        pred_mask = model(frame_rgb) #Predict mask output

                    elapsed_time_arry.append(time() - start_time) #Inference speed recorded

                    binary_mask = ((((pred_mask.squeeze() > 0.5).detach()).cpu()).numpy()).astype('uint8') #Binary mask output

                    Draw_Bbox_Seg_Single(binary_mask = binary_mask, img_bgr = frame_bgr, \
                                         defect_list = defect_list, color_arry = color_arry, \
                                         norm_post_scl = norm_post_scl, show_filtered = False) #Draw-on bounding-boxes


            ##### Classification #####
            elif (task == 'classification'):
                pass

            outvid_writer.write(frame_bgr) #Predict/Inference results write to output video

        vid_capin.release() #Close/Release video captured/streaming
        outvid_writer.release() #Close/Release output video writer

        print('===================================================')


        logger.info('=> Done! Output inferenced/predicted video results to target folder directory completed!!')

        logger.info('==> Total elapsed time: {} seconds'.format(sum(elapsed_time_arry)))
        logger.info('==> Average inference speed: {} seconds/image'.format((sum(elapsed_time_arry) / \
                                                                            len(elapsed_time_arry))))
