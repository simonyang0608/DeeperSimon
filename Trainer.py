#********************************************************************#
# Source: Trainer.py                                                 #
#                                                                    #
# Description: Major/Main loop trainer for model training/validation #
#                                                                    #
# Author: SimonYang                                                  #
#********************************************************************#

#================#
# Import Section #
#================#
####################
#Pytorch dataloader
from Data.Dataset_Mapper import DataLoader

#################################################################
#Pytorch nn module, optimizer, lr scheduler (i.e. basic inherit)
from Optimizer.Optimizer import Module, Optimizer
from Optimizer.LR_Scheduler import _LRScheduler

################
#Pytorch device
from torch import device

#######################
#Pytorch no grad, save
from torch import no_grad, save

#################################
#Measured metric, rest functions
from Utility.Metric import (Binary_Pixel_Accuracy, AUROC_Score, Dice_Score, \
                            F1_Score, AUROC_Curve, Confusion_Matrix, \
                            Over_UnderKill_Matrix, Precision_Recall, array)

#################################
#Tensorboard, data visualization
from Utility.Ploter import (SummaryWriter, Plot_AUROC_Curve, Plot_Confusion_Matrix, \
                            Add_Scalar, Add_Scalars, Add_Image, Add_Graph, \
                            Add_Figure)

##############################
#Post-process, rest functions
from Utility.PostProcessor import (Mask_To_Bbox_Multi, Mask_To_Bbox_Single, \
                                   Filter_Target, Filter_Outputs)

###################
#Tqdm progress bar
from tqdm import tqdm

############################
#Operating system (i.e. OS)
from os.path import join

####################
#Typing format list
from Utility.Metric import Any, List


#=====================#
# Class Function List #
#=====================#
class Major_Loop_Trainer(object):

    ##########################
    # Constructor Initialize #
    ##########################
    def __init__(self, cfg: dict, logger: Any, num_classes: int) -> None:
        #-----------------------------------------------#
        # Description: Constructor initialize/setup     #
        # Input type:                                   #
        #   - dict (whole config information)           #
        #   - Any (logging record)                      #
        #   - int (number of labeled classes)           #
        # Return type:                                  #
        #   - None (void, no return)                    #
        #-----------------------------------------------#

        ############
        #Initialize
        ##### Task type #####
        self.task = cfg['MODEL']['TASK']

        ##### Epochs/Iterations #####
        self.epochs = cfg['epochs']
        self.start_epoch = cfg['start_epoch']

        ##### Evaluate period #####
        self.eval_period = cfg['EVALUATE_PERIOD']

        ##### Logger/Logging #####
        self.logger = logger

        ##### Number of classes #####
        self.num_classes = num_classes

        ##### Normalized filtered-scales (i.e. camera, reference, post-process) #####
        self.camr_scl = cfg['INPUT']['CAMERA_SCALE']

        self.ref_scl = cfg['INPUT']['FILTERED_SCALE']['TRAIN_VAL']['REFERENCE']
        self.post_scl = cfg['INPUT']['FILTERED_SCALE']['TRAIN_VAL']['POST_PROCESS']

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        self.norm_ref_scl1_w, self.norm_ref_scl1_h = ((self.ref_scl)[0][0] / self.camr_scl), \
                                                     ((self.ref_scl)[0][1] / self.camr_scl)
        self.norm_ref_scl2_w, self.norm_ref_scl2_h = ((self.ref_scl)[1][0] / self.camr_scl), \
                                                     ((self.ref_scl)[1][1] / self.camr_scl)
        self.norm_ref_scl3_w, self.norm_ref_scl3_h = ((self.ref_scl)[2][0] / self.camr_scl), \
                                                     ((self.ref_scl)[2][1] / self.camr_scl)

        self.norm_post_scl1_w, self.norm_post_scl1_h = ((self.post_scl)[0][0] / self.camr_scl), \
                                                       ((self.post_scl)[0][1] / self.camr_scl)
        self.norm_post_scl2_w, self.norm_post_scl2_h = ((self.post_scl)[1][0] / self.camr_scl), \
                                                       ((self.post_scl)[1][1] / self.camr_scl)
        self.norm_post_scl3_w, self.norm_post_scl3_h = ((self.post_scl)[2][0] / self.camr_scl), \
                                                       ((self.post_scl)[2][1] / self.camr_scl)

        ##### Filtered-outputs/target (i.e. faster r-cnn detection) #####
        self.filt_outputs = []
        self.filt_target = []

        ##### Record maximun/minimun losses, info. (i.e. FCOS, MemSeg masks) #####
        self.record_max_loss = -2.0
        self.record_min_loss = 2.0

        self.record_max_info = [-2.0]
        self.record_min_info = [2.0]

        ##### Metric array (i.e. un-filtered/filtered) #####
        self.gt_arry = []
        self.filt_gt_arry = []

        self.gt_bbox_arry = []
        self.filt_gt_bbox_arry = []

        self.pred_arry = []
        self.filt_pred_arry = []

        self.pred_bbox_arry = []
        self.filt_pred_bbox_arry = []

        ##### Best accuracy #####
        self.best_accuracy = -1.


    ########################
    # Member Function List #
    ########################
    def loop_trainer(self, train_loader: DataLoader, val_loader: DataLoader, \
                     model: Module, criterion: Module, optimizer: Optimizer, \
                     scheduler: _LRScheduler, device: device, fpath: str) -> None:
        #---------------------------------------------------#
        # Description: Loop train/validate per-epoch        #
        # Input type:                                       #
        #   - DataLoader (train dataset loader)             #
        #   - DataLoader (validate dataset loader)          #
        #   - Module (self-defined model)                   #
        #   - Module (self-defined criterion/loss function) #
        #   - Optimizer (self-defined optimizer)            #
        #   - _LRScheduler (self-defined lr scheduler)      #
        #   - device (gpu/cpu device)                       #
        #   - str (file path)                               #
        # Return type:                                      #
        #   - None (void, no return)                        #
        #---------------------------------------------------#

        ############
        #Initialize
        ##### Task #####
        task = self.task

        ##### Epochs/Iterations #####
        epochs = self.epochs
        start_epoch = self.start_epoch

        ##### Evaluate period #####
        eval_period = self.eval_period

        ##### Logger/Logging #####
        logger = self.logger

        ##### Number of classes #####
        num_classes = self.num_classes

        ##### Normalized filtered-scales (i.e. square, rectangle-1, rectangle-2) #####
        norm_ref_scl = [[self.norm_ref_scl1_w, self.norm_ref_scl1_h], \
                        [self.norm_ref_scl2_w, self.norm_ref_scl2_h], \
                        [self.norm_ref_scl3_w, self.norm_ref_scl3_h]]
        
        norm_post_scl = [[self.norm_post_scl1_w, self.norm_post_scl1_h], \
                         [self.norm_post_scl2_w, self.norm_post_scl2_h], \
                         [self.norm_post_scl3_w, self.norm_post_scl3_h]]

        ##### Metric array (i.e. un-filtered/filtered) #####
        gt_arry = self.gt_arry
        filt_gt_arry = self.filt_gt_arry

        gt_bbox_arry = self.gt_bbox_arry
        filt_gt_bbox_arry = self.filt_gt_bbox_arry

        pred_arry = self.pred_arry
        filt_pred_arry = self.filt_pred_arry

        pred_bbox_arry = self.pred_bbox_arry
        filt_pred_bbox_arry = self.filt_pred_bbox_arry

        ##### Best accuracy #####
        best_accuracy = self.best_accuracy

        ##### Visualized writer #####
        writer = SummaryWriter(fpath)


        ####################
        #Whole process/flow
        for epoch in range(start_epoch, epochs):

            ##### Step 1: Train #####
            self.train(train_loader = train_loader, model = model, criterion = criterion, \
                       optimizer = optimizer, device = device, task = task, \
                       logger = logger, epoch = epoch, epochs = epochs, writer = writer, \
                       num_classes = num_classes)
            
            ##### Step 2: Validate #####
            accuracy = self.validate(val_loader = val_loader, model = model, criterion = criterion, \
                                     optimizer = optimizer, device = device, task = task, logger = logger, \
                                     epoch = epoch, epochs = epochs, eval_period = eval_period, gt_arry = gt_arry, \
                                     filt_gt_arry = filt_gt_arry, gt_bbox_arry = gt_bbox_arry, filt_gt_bbox_arry = \
                                     filt_gt_bbox_arry, pred_arry = pred_arry, filt_pred_arry = filt_pred_arry, \
                                     pred_bbox_arry = pred_bbox_arry, filt_pred_bbox_arry = filt_pred_bbox_arry, \
                                     writer = writer, num_classes = num_classes, norm_ref_scl = norm_ref_scl, \
                                     norm_post_scl = norm_post_scl)

            ##### Step 3: Scheduler step #####
            Add_Scalar(writer = writer, title = 'Learning_Rate', value = scheduler.get_lr(), \
                       epoch = epoch)

            scheduler.step()

            ##### Step 4: Save checkpoint #####
            if (((epoch + 1) % eval_period) == 0):
                self.save_checkpoint(state = {'epoch': (epoch + 1), \
                                              'state_dict': model.state_dict(), \
                                              'optimizer': optimizer.state_dict(), \
                                              'scheduler': scheduler.state_dict()}, \
                                     fpath = fpath) #Save at evaluate epoch/iteration interval
            else:
                pass
            
            if (epoch > (0.9 * epochs)): #Last epoch/iteration interval
                if (accuracy >= best_accuracy): #Best accuracy updated
                    best_accuracy = accuracy

                    self.save_checkpoint(state = {'epoch': (epoch + 1), \
                                                  'state_dict': model.state_dict(), \
                                                  'optimizer': optimizer.state_dict(), \
                                                  'scheduler': scheduler.state_dict()}, \
                                         fpath = fpath, is_best = True) #Save at last epoch/iteration interval
                else:
                    pass
            else:
                pass

        writer.close() #Close/Release visualized writer


    def train(self, train_loader: DataLoader, model: Module, \
              criterion: Module, optimizer: Optimizer, device: device, \
              task: str, logger: Any, epoch: int, epochs: int, \
              writer: SummaryWriter, num_classes: int) -> None:
        #---------------------------------------------------#
        # Description: Train per-epoch                      #
        # Input type:                                       #
        #   - Dataloader (train dataset loader)             #
        #   - Module (self-defined model)                   #
        #   - Module (self-defined criterion/loss function) #
        #   - Optimizer (self-defined optimizer)            #
        #   - device (gpu/cpu device)                       #
        #   - str (tasks type)                              #
        #   - Any (logging record)                          #
        #   - int (epoch/iteration)                         #
        #   - int (total/whole epochs/iterations)           #
        #   - SummaryWriter (visualized data writer)        #
        #   - int (number of labeled classes)               #
        # Return type:                                      #
        #   - None (void, no return)                        #
        #---------------------------------------------------#

        ############
        #Initialize
        ##### Switch model to train mode #####
        model.train()

        ##### Train with tqdm progress-bar #####
        t_epoch = tqdm(train_loader)

        t_epoch.set_description("Epoch [{}/{}] -> Train" \
                                .format(epoch, (epochs - 1))) #Set tqdm description
    

        ##############################################
        #Customized train status with different tasks
        ##### Detection #####
        if (task == 'detection'):
            for (image, target) in t_epoch:
                image = list(sub_image.to(device) for \
                             sub_image in image) #Re-create aligned batch images list

                target = list({sub_key: sub_value.to(device) for \
                               sub_key, sub_value in sub_target.items()} for \
                               sub_target in target) #Re-create aligned batch targets list
                
                outputs = model(image, target) #Loss output
                
                loss_value = sum(sub_loss for sub_loss in outputs.values()) #Loss value calculate

                rpn_cls_loss, rpn_bbox_loss, total_loss = \
                                    float(outputs['loss_objectness']), \
                                    float(outputs['loss_rpn_box_reg']), \
                                    float(loss_value)
                
                optimizer.zero_grad() #Pre-freeze gradient
                loss_value.backward() #Back-propagation pass
                optimizer.step() #Gradient update

                t_epoch.set_postfix(rpn_cls_loss = rpn_cls_loss, rpn_bbox_loss = rpn_bbox_loss, \
                                    total_loss = total_loss)
                
            Add_Scalar(writer = writer, title = 'RPN_Class_Loss/Train', value = \
                       rpn_cls_loss, epoch = epoch) #Visualized rpn class loss curve
                
            Add_Scalar(writer = writer, title = 'RPN_Bbox_Loss/Train', value = \
                       rpn_bbox_loss, epoch = epoch) #Visualized rpn bbox loss curve
            
            Add_Scalar(writer = writer, title = 'Total_Loss/Train', value = \
                       total_loss, epoch = epoch) #Visualized total loss curve
                
            logger.info('Epoch [{}/{}] -> Train: rpn_cls_loss: {}  rpn_bbox_loss: {}  total_loss: {}' \
                        .format(epoch, epochs, rpn_cls_loss, rpn_bbox_loss, total_loss))


        ##### Segmentation #####
        elif (task == 'segmentation'):
            for (image, mask) in t_epoch:
                image = image.to(device) #Point image data to main device
                mask = mask.to(device) #Point mask data to main device
                
                if (num_classes > 1): #Multiple classes
                    pred_mask, pred_argmax = model(image) #Predict mask output
                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = pred_argmax, target_mask = mask) #Accuracy value calculate

                else: #Single class
                    pred_mask = model(image) #Predict mask output

                    binary_mask = (pred_mask > 0.5).float() #Binary mask output
                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = binary_mask, target_mask = mask) #Accuracy value calculate

                loss_value = (criterion(pred_mask, mask)).float() #Loss value calculate

                optimizer.zero_grad() #Pre-freeze gradient
                loss_value.backward() #Back-propagation pass
                optimizer.step() #Gradient update

                t_epoch.set_postfix(loss = loss_value.item(), accuracy = (100. * accuracy_value))

            Add_Scalar(writer = writer, title = 'Accuracy/Train', value = accuracy_value, \
                       epoch = epoch) #Visualized accuracy curve
                
            Add_Scalar(writer = writer, title = 'Loss/Train', value = loss_value, \
                       epoch = epoch) #Visualized loss curve

            logger.info('Epoch [{}/{}] -> Train: accuracy: {}  loss: {}'.format(epoch, epochs, \
                       (100. * accuracy_value), loss_value.item()))


        ##### Classification #####
        elif (task == 'classification'):
            pass


    def validate(self, val_loader: DataLoader, model: Module, criterion: Module, \
                 optimizer: Optimizer, device: device, task: str, logger: Any, \
                 epoch: int, epochs: int, eval_period: int, gt_arry: List, \
                 filt_gt_arry: List, gt_bbox_arry: List, filt_gt_bbox_arry: List, \
                 pred_arry: List, filt_pred_arry: List, pred_bbox_arry: List, \
                 filt_pred_bbox_arry: List, writer: SummaryWriter, num_classes: int, \
                 norm_ref_scl: list, norm_post_scl: list) -> float:
        #---------------------------------------------------------------#
        # Description: Validate per-epoch                               #
        # Input type:                                                   #
        #   - Dataloader (validate dataset loader)                      #
        #   - Module (self-defined model)                               #
        #   - Module (self-defined criterion/loss function)             #
        #   - Optimizer (self-defined optimizer)                        #
        #   - device (gpu/cpu device)                                   #
        #   - str (tasks type)                                          #
        #   - Any (logging record)                                      #
        #   - int (epoch/iteration)                                     #
        #   - int (total/whole epochs/iterations)                       #
        #   - int (validate/evaluate epoch/iteration)                   #
        #   - List (ground-truth array)                                 #
        #   - List (filtered ground-truth array)                        #
        #   - List (ground-truth bounding-boxes array)                  #
        #   - List (filtered ground-truth bounding-boxes array)         #
        #   - List (predict result array)                               #
        #   - List (filtered predict result array)                      #
        #   - List (predict result bounding-boxes array)                #
        #   - List (filtered predict result bounding-boxes array)       #
        #   - SummaryWriter (visualized data writer)                    #
        #   - int (number of labeled classes)                           #
        #   - list (normalized filtered-scale list (i.e. reference))    #
        #   - list (normalized filtered-scale list (i.e. post-process)) #
        # Return type:                                                  #
        #   - float (result accuracy value)                             #
        #---------------------------------------------------------------#

        ############
        #Initialize
        ##### Switch model to suitable/fit mode #####
        if ((task == 'segmentation') or (task == 'classification')): #Segmentation/Classification
            model.eval()

        else: #Detection
            model.train()

        ##### Validate with tqdm progress-bar #####
        t_epoch = tqdm(val_loader)

        t_epoch.set_description("Epoch [{}/{}] -> Validate" \
                                .format(epoch, (epochs - 1))) #Set tqdm description


        #################################################
        #Customized validate status with different tasks
        ##### Detection #####
        if (task == 'detection'):
            for (image, target) in t_epoch:
                image = list(sub_image.to(device) for \
                             sub_image in image) #Re-create aligned batch images list

                target = list({sub_key: sub_value.to(device) for \
                               sub_key, sub_value in sub_target.items()} for \
                               sub_target in target) #Re-create aligned batch targets list
                
                outputs = model(image, target) #Loss output
                
                loss_value = sum(sub_loss for sub_loss in outputs.values()) #Loss value calculate

                rpn_cls_loss, rpn_bbox_loss, total_loss = \
                                    float(outputs['loss_objectness']), \
                                    float(outputs['loss_rpn_box_reg']), \
                                    float(loss_value)
                
                optimizer.zero_grad() #Pre-freeze gradient

                t_epoch.set_postfix(rpn_cls_loss = rpn_cls_loss, rpn_bbox_loss = rpn_bbox_loss, \
                                    total_loss = total_loss)

                if (((epoch + 1) % eval_period) == 0): #Store output results from batch images
                    model.eval() #Switch model to validate mode

                    with no_grad():
                        outputs = model(image, None) #Predict labels, bboxes, scores output

                    Filter_Target(norm_ref_scl = norm_ref_scl, target = target, filt_target = self.filt_target) #Filtered-out target list
                    Filter_Outputs(norm_post_scl = norm_post_scl, outputs = outputs, filt_outputs = self.filt_outputs) #Filtered-out outputs list

                    gt_arry.extend(target) #Predict tensors extensions
                    pred_arry.extend(outputs) #Ground-truth tensors extensions

                    filt_gt_arry.extend(self.filt_target) #Filtered-predict tensors extensions
                    filt_pred_arry.extend(self.filt_outputs) #Filtered ground-truth tensors extensions

                    model.train() #Switch model to train mode

                    if (self.filt_target):
                        (self.filt_target).clear() #Reset
                    else:
                        pass

                    if (self.filt_outputs):
                        (self.filt_outputs).clear() #Reset
                    else:
                        pass

                else:
                    pass

            Add_Scalar(writer = writer, title = 'RPN_Class_Loss/Validate', value = \
                       rpn_cls_loss, epoch = epoch) #Visualized rpn class loss curve
                
            Add_Scalar(writer = writer, title = 'RPN_Bbox_Loss/Validate', value = \
                       rpn_bbox_loss, epoch = epoch) #Visualized rpn bbox loss curve
            
            Add_Scalar(writer = writer, title = 'Total_Loss/Validate', value = \
                       total_loss, epoch = epoch) #Visualized total loss curve
                
            
            ##### Visualized metrics from output results #####
            if (((epoch + 1) % eval_period) == 0):
                try:
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
                                'Filtered': filt_precision_50}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Precision/75', dict_value = {'Un-Filtered': precision_75, \
                                'Filtered': filt_precision_75}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'Precision/85', dict_value = {'Un-Filtered': precision_85, \
                                'Filtered': filt_precision_85}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'Recall/50', dict_value = {'Un-Filtered': recall_50, \
                                'Filtered': filt_recall_50}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Recall/75', dict_value = {'Un-Filtered': recall_75, \
                                'Filtered': filt_recall_75}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Recall/85', dict_value = {'Un-Filtered': recall_85, \
                                'Filtered': filt_recall_85}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'F1_Score/50', dict_value = {'Un-Filtered': f1_score_50, \
                                'Filtered': filt_f1_score_50}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'F1_Score/75', dict_value = {'Un-Filtered': f1_score_75, \
                                'Filtered': filt_f1_score_75}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'F1_Score/85', dict_value = {'Un-Filtered': f1_score_85, \
                                'Filtered': filt_f1_score_85}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)
                
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_50), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_50), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                    
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_75), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_75), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                    
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_85), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_85), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_50), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.5)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_50), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.5)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_75), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.75)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_75), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.75)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_85), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.85)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_85), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.85)

                except Exception as _:
                    pass

                finally:
                    gt_arry.clear() #Reset
                    filt_gt_arry.clear()

                    pred_arry.clear() #Reset
                    filt_pred_arry.clear()

            else:
                pass
                
            logger.info('Epoch [{}/{}] -> Validate: rpn_cls_loss: {}  rpn_bbox_loss: {}  total_loss: {}' \
                        .format(epoch, epochs, rpn_cls_loss, rpn_bbox_loss, total_loss))
            
            print('===================================================')

            return (1. - total_loss)


        ##### Segmentation #####
        elif (task == 'segmentation'):
            for (image, mask) in t_epoch:
                image = image.to(device) #Point image data to main device
                mask = mask.to(device) #Point mask data to main device

                if (num_classes > 1): #Multiple classes
                    with no_grad():
                        pred_mask, pred_argmax = model(image) #Predict mask output

                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = pred_argmax, target_mask = mask) #Accuracy value calculate

                else: #Single class
                    with no_grad():
                        pred_mask = model(image) #Predict mask output

                    binary_mask = (pred_mask > 0.5).float() #Binary mask output
                    accuracy_value = Binary_Pixel_Accuracy(binary_mask = binary_mask, target_mask = mask) #Accuracy value calculate

                loss_value = (criterion(pred_mask, mask)).float() #Loss value calculate

                t_epoch.set_postfix(loss = loss_value.item(), accuracy = (100. * accuracy_value))

                if (((epoch + 1) % eval_period) == 0): #Store output results from batch images
                    batch_range = (image.shape)[0] #Batch range sizes
                    
                    if (num_classes > 1): #Multiple classes
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

                        if (loss_value > self.record_max_loss): #Track for maximun loss
                            self.record_max_loss = loss_value

                            (self.record_max_info).clear() #Reset

                            (self.record_max_info).append(image)
                            (self.record_max_info).append(((mask.unsqueeze(1)) / (num_classes - 1)))
                            (self.record_max_info).append(((pred_argmax.unsqueeze(1)) / (num_classes - 1)))
                        else:
                            pass
                                
                        if (loss_value < self.record_min_loss): #Track for minimun loss
                            self.record_min_loss = loss_value

                            (self.record_min_info).clear() #Reset

                            (self.record_min_info).append(image)
                            (self.record_min_info).append((mask.unsqueeze(1)) / (num_classes - 1))
                            (self.record_min_info).append((pred_argmax.unsqueeze(1)) / (num_classes - 1))
                        else:
                            pass

                        pred_arry.extend((pred_argmax.view(-1)).tolist())

                    else: #Single class
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

                        if (loss_value > self.record_max_loss): #Track for maximun loss
                            self.record_max_loss = loss_value

                            (self.record_max_info).clear() #Reset

                            (self.record_max_info).append(image)
                            (self.record_max_info).append(mask)
                            (self.record_max_info).append(binary_mask)
                        else:
                            pass

                        if (loss_value < self.record_min_loss): #Track for minimun loss
                            self.record_min_loss = loss_value

                            (self.record_min_info).clear() #Reset

                            (self.record_min_info).append(image)
                            (self.record_min_info).append(mask)
                            (self.record_min_info).append(binary_mask)
                        else:
                            pass

                        pred_arry.extend((binary_mask.view(-1)).tolist())

                    gt_arry.extend((mask.view(-1)).tolist())

                else:
                    pass

            Add_Scalar(writer = writer, title = 'Accuracy/Validate', value = accuracy_value, \
                       epoch = epoch) #Visualized accuracy curve
                
            Add_Scalar(writer = writer, title = 'Loss/Validate', value = loss_value, \
                       epoch = epoch) #Visualized loss curve


            ##### Visualized model architecture/metrics from output results #####
            if (not epoch):
                Add_Graph(writer = writer, model = model, image = image) #Neuron/Model architecture graph visualized
            else:
                pass

            if (((epoch + 1) % eval_period) == 0):
                if (len(set(gt_arry)) != 1):
                    pred_np_arry, gt_np_arry = array(pred_arry), array(gt_arry) #Numpy array conversions (i.e. predict, ground-truth)

                    if (num_classes == 1): #Single class
                        dice_score_value = Dice_Score(predict = pred_np_arry, ground_truth = gt_np_arry) #Dice score
                        auroc_score_value = AUROC_Score(predict = pred_np_arry, ground_truth = gt_np_arry) #AUROC score
                        fp_rate, tp_rate, _ = AUROC_Curve(predict = pred_np_arry, ground_truth = gt_np_arry) #AUROC curve

                        Add_Scalar(writer = writer, title = 'Dice_Score', value = dice_score_value, \
                                   epoch = epoch) #Dice score curve visualized

                        Add_Scalar(writer = writer, title = 'AUROC_Score', value = auroc_score_value, \
                                   epoch = epoch) #AUROC score curve visualized

                        Add_Figure(writer = writer, title = 'AUROC_Curve', figure = Plot_AUROC_Curve(fp_rate = fp_rate, \
                                   tp_rate = tp_rate, auroc_score = auroc_score_value), epoch = epoch) #AUROC area curve visualized
                        
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
                                'Filtered': filt_precision_50}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Precision/75', dict_value = {'Un-Filtered': precision_75, \
                                'Filtered': filt_precision_75}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'Precision/85', dict_value = {'Un-Filtered': precision_85, \
                                'Filtered': filt_precision_85}, epoch = epoch) #Precision score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'Recall/50', dict_value = {'Un-Filtered': recall_50, \
                                'Filtered': filt_recall_50}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Recall/75', dict_value = {'Un-Filtered': recall_75, \
                                'Filtered': filt_recall_75}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'Recall/85', dict_value = {'Un-Filtered': recall_85, \
                                'Filtered': filt_recall_85}, epoch = epoch) #Recall score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                    Add_Scalars(writer = writer, title = 'F1_Score/50', dict_value = {'Un-Filtered': f1_score_50, \
                                'Filtered': filt_f1_score_50}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.5, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'F1_Score/75', dict_value = {'Un-Filtered': f1_score_75, \
                                'Filtered': filt_f1_score_75}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.75, un-filtered/filtered)
                    
                    Add_Scalars(writer = writer, title = 'F1_Score/85', dict_value = {'Un-Filtered': f1_score_85, \
                                'Filtered': filt_f1_score_85}, epoch = epoch) #F1 score curve visualized (i.e. iou threshold = 0.85, un-filtered/filtered)

                    Add_Figure(writer = writer, title = 'Pixel_based_Confusion_Matrix', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix), epoch = epoch) #Confusion matrix visualized (i.e. pixel-based)
                
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_50), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_50), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.5)
                    
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_75), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_75), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.75)
                    
                    Add_Figure(writer = writer, title = 'Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               over_underkill_matrix_85), epoch = epoch) #Over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                    Add_Figure(writer = writer, title = 'Filtered-Over_UnderKill_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_over_underkill_matrix_85), epoch = epoch) #Filtered over/Under-kill matrix visualized (i.e. iou threshold = 0.85)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_50), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.5)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/50', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_50), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.5)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_75), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.75)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/75', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_75), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.75)
                    
                    Add_Figure(writer = writer, title = 'Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               confusion_matrix_85), epoch = epoch) #Confusion matrix visualized (i.e. iou threshold = 0.85)
                    Add_Figure(writer = writer, title = 'Filtered-Confusion_Matrix/85', figure = Plot_Confusion_Matrix(confusion_matrix = \
                               filt_confusion_matrix_85), epoch = epoch) #Filtered-confusion matrix visualized (i.e. iou threshold = 0.85)

                else:
                    pass

                Add_Image(writer = writer, title = 'High_Loss/Images', image = (self.record_max_info)[0], \
                          epoch = epoch) #High loss batch images visualized
                Add_Image(writer = writer, title = 'Low_Loss/Images', image = (self.record_min_info)[0], \
                          epoch = epoch) #Low loss batch images visualized
                
                Add_Image(writer = writer, title = 'High_Loss/Target_Masks', image = (self.record_max_info)[1], \
                          epoch = epoch) #High loss batch target masks visualized
                Add_Image(writer = writer, title = 'Low_Loss/Target_Masks', image = (self.record_min_info)[1], \
                          epoch = epoch) #Low loss batch target masks visualized
                
                Add_Image(writer = writer, title = 'High_Loss/Predict_Masks', image = (self.record_max_info)[2], \
                          epoch = epoch) #High loss batch predict masks visualized
                Add_Image(writer = writer, title = 'Low_Loss/Predict_Masks', image = (self.record_min_info)[2], \
                          epoch = epoch) #Low loss batch predict masks visualized
                
                self.record_max_loss = -2.0 #Reset
                self.record_min_loss = 2.0 #Reset

                gt_arry.clear() #Reset

                gt_bbox_arry.clear() #Reset
                filt_gt_bbox_arry.clear()

                pred_arry.clear() #Reset

                pred_bbox_arry.clear() #Reset
                filt_pred_bbox_arry.clear()

            else:
                pass

            logger.info('Epoch [{}/{}] -> Validate: accuracy: {}  loss: {}'.format(epoch, epochs, \
                    (100. * accuracy_value), loss_value.item()))
        
            print('===================================================')

            return accuracy_value
        

        ##### Classification #####
        elif (task == 'classification'):
            pass
    

    def save_checkpoint(self, state: dict, fpath: str, fname: str = 'checkpoint.pth', \
                        is_best: bool = False) -> None:
        #----------------------------------------------------------#
        # Description: Save checkpoint for whole model information #
        # Input type:                                              #
        #   - dict (whole model information)                       #
        #   - str (file path)                                      #
        #   - str (file name)                                      #
        #   - bool (best accuracy/model or not)                    #
        # Return type:                                             #
        #   - None (void, no return)                               #
        #----------------------------------------------------------#

        ####################
        #Whole process/flow
        ##### Step 1: Save checkpoint at each epoch/iteration #####
        save(state, join(fpath, fname))

        ##### Step 2: Save best model at last epoch/iteration interval #####
        if (is_best):
            save(state, join(fpath, ('best_' + fname)))
        
        else:
            pass