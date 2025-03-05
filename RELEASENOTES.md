# Release Notes

## 2024/06/14 v1.7.0
- New Features
    1. [Modify] Modify for the correct measurement/methods of over/under-kill confusion matrix generations
    2. [Modify] Modify for adding the external measurement/methods of multi-classes confusion matrix towards over/under-kill matrix
    3. [Modify] Modify for subtle changes with code alignment
    4. [Modify] Modify for subtle changes with code alignment
    5. [Modify] Modify for subtle changes with code alignment
    6. [Modify] Modify for updating the non-maximun mechanismns into evaluations and metrics, also avoid/fix for the invalid f1-score values
    7. [Modify] Modify for update/synchornize the defects combined in same region and masks to bounding-boxes conversions functions
    8. [Modify] Modify for subtle changes from start epoch to current epochs during metric measurements while model training/validation
    9. [Modify] Modify for vovnetv2 model layer blocks from depth-wise seperable to fully/normal convolutions operations
    10. [Modify] Modify for self-defined defect definitions documents/file in argument parser instead of fixed info.
    11. [Modify] Modify for updating install markdown file (i.e. sync virtual environment installations)
    12. [Add] Add/Prepare the required images for install markdown file
    13. [Modify] Modify for the smooth-process convolutions/operations from 1x1 to 3x3 kernel size for meta network/models
    14. [Modify] Modify for coding-style more briefer/ alignment

## 2024/05/13 v1.6.0
- New Features
    1. [Modify] Modify for coding-style alignment, sync (i.e. subtle change)
    2. [Modify] Modify for the aspect ratio range of faster r-cnn model
    3. [Modify] Modify for the NMS (i.e. non-maximun suppression) threshold of object detection models during evaluations
    4. [Modify] Modify for split model status to train, evaluate, and export mode with different process
    5. [Modify] MOdify for tensor post-process from object detection models outputs
    6. [Modify] Modify for subtle changes with code alignment
    7. [Modify] Modify for subtle changes with code alignment and tensor post-process during evaluations
    8. [Modify] Modify for clearer/briefer operations of model training/evaluation loop code
    9. [Modify] Modify for dataset augmentation process/flow with less time and spatial complexity
    10. [Modify] Modify for threshold values defined in non-maximun suppression process/flow
    11. [Modify] Modify for coding-style more briefer/ alignment
    12. [Modify] Modify for random augments dataset in batched-unit instead of whole quantities brutally
    13. [Modify] Modify for updating the class distributuins/names in confusion matrix settings/arguments
    14. [Modify] Modify for coding-style more briefer/ alignment
    15. [Modify] Modify for updating the measurements of precision, recall, and f1-score value metrics
    16. [Modify] Modify for updating the measurements of precision, recall, and f1-score values, and over/under-kill matrix based on bbox iou for object-detection models and ploter
    17. [Modify] Modify for clearer descriptions in return types from dataset preprocessor
    18. [Modify] Modify for briefer, clearer process/flow of exportions, with subtle changes in input tensors fit for general detection tasks

## 2024/03/12 v1.5.0
- New Features
    1. [Modify] Modify for argmax conversions for multi-classes array before memory copy from cpu to gpu
    2. [Modify] Remove softmax conversions for redundant confidence measurement output from model
    3. [Modify] Rename shell script file name for training pipeline generation/preparation
    4. [Modify] Modify layout/type-settings a little bit
    5. [Modify] Modify for variable defined error/invalid in loss function code script
    6. [Modify] Modify for simple re-factor coding style for the following code scripts
    7. [Modify] Modify dataset preprocess, documents related scripts for the required object detection task
    8. [Modify] Modify model construct/re-factor flow for required object detection task (i.e. faster r-cnn, ...)
    9. [Modify] Modify the model train/validate optimization flow for the required object detection task
    10. [Modify] Modify for subtle changes in related scripts to fit for the required object detection task
    11. [Modify] Modify for able to filtering out the images/objects which didn't need to be detected
    12. [Modify] Modify for able to train/validate pure background/negative samples without bboxes, ...
    13. [Modify] Modify for samples/tensors data processing & metrics measurement during training/validation
    14. [Modify] Modify for able to set resize scale parameters in object detection models
    15. [Modify] Modify for able to evaluate the performances of object detection models from dataset/images
    16. [Modify] Modify for clearer and brifer remark, code, ...
    17. [Modify] Modify for clearer progress-bar descriptions during evaluation
    18. [Modify] Modify for able to export the defined-format from object detection models

## 2023/11/21 v1.4.0
- New Features
    1. [Modify] Modify the related functions/requirements for previous stage
    2. [Modify] Modify readme markdown file for layout a little bit
    3. [Modify] Modify the related functions/requirements for previous stage
    4. [Modify] Modify the related functions/requirements for previous stage
    5. [Modify] Sync sigmoid function into forward pass of model
    6. [Modify] Sync brightness augmented factors into data preprocess flow
    7. [Modify] Modified sync brightness augmented factors for further suitable values
    8. [Modify] Modified sync brightness augmented factors for further suitable values updated
    9. [Modify] Sync the single/multi-classes segmentation flow into main system/program based on self-defined defect classes
    10. [Modify] Sync the general training pipeline into main script/program
    11. [Modify] Modify for subtle changes of returned variables

## 2023/9/20 v1.3.0
- New Features
    1. [Add] Add shell script for main process/workflow operations
    2. [Modify] Modify install markdown file a little bit for the following samples
    3. [Add] Add/Prepare the required images for install markdown file
    4. [Modify] Modify install markdown file for the Installation process flow/pipeline verifications
    5. [Modify] Modify shell script for the main process/flow operations to match directory settings in docker file
    6. [Modify] Modify the related functions/requirements for previous stage
    7. [Modify] Modify readme markdown file for the whole main process flow/pipeline verifications for first part
    8. [Add] Add/Prepare the required images for readme markdown file
    9. [Modify] Modify the related functions/requirements for previous stage
    10. [Add] Add/Prepare the required images for readme markdown file
    11. [Modify] Modify readme markdown file for the whole main process flow/pipeline verifications for next part
    12. [Modify] Modify the related functions/requirements for previous stage
    13. [Add] Add/Prepare the required images for readme markdown file
    14. [Modify] Modify readme markdown file for the whole main process flow/pipeline verifications for final part

## 2023/8/23 v1.2.0
- New Features
    1. [Modify] Modify the related functions/requirements for previous stage
    2. [Modify] Optimizing functions development & verification
    3. [Modify] Functions for output results measured and visualized development & verification
    4. [Modify] Training/Validation process flow/pipeline development & verification
    5. [Modify] Modify the related functions/requirements for previous stage
    6. [Modify] Evaluation/Inference process flow/pipeline development & verification
    7. [Modify] Modify the related functions/requirements for previous stage
    8. [Modify] Exportion/Convert process flow/pipeline development & verification

## 2023/7/19 v1.1.0
- New Features
    1. [Modify] Modify for re-organize config folder/file distribution
    2. [Delete] Delete original organized config folder/file distribution
    3. [Modify] Dataset generate flow development & verification
    4. [Modify] Requirement packages & config yaml double-check/verify
    5. [Modify] Dataset details flow double-check
    6. [Modify] Model block generate flow development & verification
    7. [Modify] Model backbone/decoder generate flow development & verification
    8. [Modify] Whole model network/architecture generate flow development & verification
    9. [Modify] Model summarizer/measurement flow development & verification
    10. [Modify] Start for the development for next model training/validation process

## 2023/6/28 v1.0.0
- New Features
    1. [Add] add required .md (i.e. markdown), .txt, ..., and docker file
    2. [Add] add required folder, codebase file/script layout/distribution
    3. [Modify] Modify some details in related documents