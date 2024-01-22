**How to Use**

1) Create a folder named "yolov8_train_test" on Google Drive and upload all the files and folders to it.

2) Open the Google Collab after it has been uploaded to the Google Drive

3) connect the Google Drive folder "yolov8_train_test" to the Google Collab runtime

4) create a T4 runtime (GPU) to run the notebook

5) follow instructions in the notebook to successfully run the notebook cells.

6) Run all cells except "Training" if no training is required


**Output**

- All the output files are stored in the output folder

- Output/output folder contains all the masked images and mask files output by the SAM model

- Output/predict folder contains all the bounding boxes files created by YOLO V8

**Test with new Images**

- add new .jpg images to the /test_images folder and run the YOLO V8 inference and SAM inference to get the masked image. 


**Report**


**Abstract**

The manual masking of fired cartridge cases, a critical function in forensic ballistics, is a process that has traditionally been time-consuming and prone to human error. To address these challenges, this report introduces a novel algorithm designed to automate the masking and coloring of cartridge case images. This automation is particularly vital for sorting and comparing cases to determine their firearm origins, a task of paramount importance in police investigations.

**Introduction**

The examination of cartridge cases plays a vital role in forensic investigations involving firearms. Features such as breech-face impressions and firing pin impressions are unique to each weapon and can connect a piece of ammunition to its firearm origin. The precise masking of these features on cartridge case images allows forensic software to more effectively identify and compare these distinctive characteristics. The automation of this process presents an opportunity to significantly accelerate investigative workflows and enhance the accuracy of forensic analyses.

**Methodology**

![image](https://github.com/abamrah/Masking-Cartridge-Case-using-YOLO-v8-and-SAM/assets/71141583/d651309c-1615-4d05-a673-2043f17ec0fd)

The developed algorithm employs a composite machine learning approach, integrating the strengths of YOLO V8 for object detection and the Segment Anything Model (SAM) for image segmentation. YOLO V8, renowned for its efficiency and accuracy in detecting objects within images, was fine-tuned to recognize specific features on cartridge cases. This fine-tuning involved training the model to identify different feature categories relevant to forensic analysis, including breech-face impressions, aperture shears, firing pin impressions, firing pin drags, and the direction of the firing pin drag. Once these features were detected and their respective bounding boxes were established, the SAM utilized this information to create segmented masks corresponding to each feature.

A dataset from Kaggle, consisting of 9mm cartridge case images, was annotated using Label box to delineate the features of interest. Given the constraints on resources and time, a subset of 70 images underwent annotation, serving as a training set for YOLO V8. Despite the limited dataset, YOLO V8's robust learning algorithms were able to yield promising results, demonstrating the model's potential for one-shot learning and its applicability in scenarios with sparse data.

**Results**

![image](https://github.com/abamrah/Masking-Cartridge-Case-using-YOLO-v8-and-SAM/assets/71141583/4071669b-8469-43dc-899c-1bde25e305cf)


The training process of the YOLO V8 model for the automated masking algorithm reveals a robust learning pattern. The losses, including box loss, class loss, and direction of firing pin drag (dfI_loss), exhibit a downward trend as training progresses, indicative of the model’s improving accuracy in feature localization and classification. The training box loss stabilizes around 1.0, while the validation box loss converges similarly, suggesting good generalization to the validation set.

Precision and recall metrics, which are crucial indicators of the model's performance, show strong results. Precision remains consistently above 0.8, and recall trends towards 0.9, indicating that the model is highly accurate in identifying relevant features and that it retrieves most of the positive samples. The mean Average Precision (mAP) at Intersection over Union (IoU) of 0.5 (mAP50) is maintained above 0.6 throughout the training, which, combined with an mAP at IoU of 0.5 to 0.95 (mAP50-95) above 0.5, validates the model's robustness across different thresholds of IoU.

**Discussion**

The training and validation losses present a compelling narrative about the model's capacity to learn and adapt to the task of identifying and masking features on cartridge cases. The convergence of training and validation losses implies that the model is not overfitting and is expected to perform well on unseen data.

The high precision metric is particularly significant, as it indicates a low false positive rate — an essential factor in forensic analysis where the cost of misidentification is high. The high recall metric suggests that the model has a low false negative rate, which is equally important in ensuring that no critical evidence is missed.

The mAP scores are benchmarks for the overall quality of the model across all classes and IoU thresholds. An mAP50 above 0.6 signifies that the model is highly accurate when the IoU threshold is 0.5, a common benchmark in object detection tasks. The mAP50-95, which averages mAP across IoU thresholds from 0.5 to 0.95, being above 0.5, indicates that the model is robust even when stricter localization criteria are applied. This is particularly useful in forensic applications where the precise delineation of features is necessary.

The observed metrics demonstrate that the algorithm is robust and reliable, making it a suitable candidate for automating the task of masking cartridge cases in a forensic setting. The performance could potentially be improved even further by expanding the training dataset and continuing to refine the model parameters.

In conclusion, the algorithm exhibits a strong capability for accurately identifying and masking the pertinent features of cartridge cases. Its implementation in a forensic context could significantly streamline the analysis process, offering not only time savings but also enhancing the reliability of the results which are critical for police investigations.

**Conclusion**

The implementation of an automated masking algorithm represents a significant technological advancement in the field of forensic ballistics. By harnessing the capabilities of advanced machine learning models, the algorithm facilitates the rapid and accurate analysis of cartridge cases, which is essential for the timely resolution of police investigations. As the algorithm continues to evolve through further research and development, it holds the promise of becoming a standard tool in forensic laboratories worldwide, offering a high degree of reliability and efficiency in ballistic examinations.

**References**

"Kaggle Dataset for Bullet Cartridges." Kaggle. [Online]. Available: https://www.kaggle.com/datasets/dnnpy1/tracks-on-bullet-cartridges

"Labelbox: Open Source Labeling Tool." Labelbox. [Online]. Available: https://labelbox.com/

"YOLOv8: Train Custom Data." Ultralytics. [Online]. Available: https://docs.ultralytics.com/modes/train/

"Segment Anything Model (SAM) GitHub Repository." [Online]. Available: https://github.com/facebookresearch/segment-anything

Result

**Output YOLO and SAM**

![image](https://github.com/abamrah/Masking-Cartridge-Case-using-YOLO-v8-and-SAM/assets/71141583/4727b39f-10d3-4d76-8f6c-23e64dc8b756)

![image](https://github.com/abamrah/Masking-Cartridge-Case-using-YOLO-v8-and-SAM/assets/71141583/e9dc9b61-2781-4f93-8ec2-98c0c302343d)
