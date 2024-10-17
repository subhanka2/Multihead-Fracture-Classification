# qMSK-Multihead-Attention
Large Scale Validation of a Multi-Head Deep Learning Model for Fracture Detection and Localization across Various Anatomical Regions

# Model Architecture
The model architecture incorporates the EfficientNetV2 encoder for efficient feature extraction. The encoder is complemented by a multi-task approach, featuring 19 spinal-net classification heads and 1 fracture segmentation head, which is constructed based on the Unet++ decoder. This combination allows our model to excel in both classification and precise fracture segmentation tasks.

![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/MSK_FINA.png)

# Results

AUC-ROC Plot: Body part Detection on Test Dataset: (A) Upper Limb (B) Lower Limb
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/testing_internal_data2.png)

AUC-ROC Plot: Fracture Detection on Test dataset: (A) Upper Limb (B) Lower Limb
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/testing_internal_data2_fracture.png)

AUC-ROC Plot: Treated fracture Detection on Test Dataset: (A) Upper Limb (B) Lower Limb
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/treated_internal_data2_fracture.png)

AUC-ROC Plot: Fracture Detection on Erasmus University Medical Center dataset: (A) Upper Limb (B) Lower Limb
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/Erasmus_plot3.png)

AUC-ROC Plot: Fracture Detection on Open-source Validation Datasets: (A) FracAtlas (B) GRAZPEDWRI-DX
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/Open_source-auc.png)


Representative True Positive: Wrist and Foot X-ray with AI segmented overlay
![alt text](https://github.com/subhanka2/qMSK-Multihead-Attention/blob/main/Images/TP_QMSK.png)

