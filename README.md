# Bearing_Fault_analysis
My college's project
I was responsible for analyzing the signal and designed Machine Learning algorithm (Pic 3).  First, I transformed the signal of bearing vibration from time domain to frequency domain.  Second, I used Isolation Forest to detect outlier because our sensor easily received noise.  Third, I used Random Forest as my classifier.  I could easily get the feature importance by Mean Decrease Impurity and Permutation Importance to determine which frequency was important to my model.  By feature selection and pruning, I could restrict the complexity of the model.  Because its structure was simple, so my classmate could easily speed up this algorithm parallelly on PYNQ-Z2 board under limited resource.  At the end, we got first place in 創創 AIoT Hackathon team competition 2020.

<img src="/home/aaon/文件/assets/1543803099995.png" style="zoom:70%" />

## Read file 
Please use ./Header/preprocessing.py/read_file(...)

## Feature Selection
Please use ./Header/model_selection/GetFeatureImportanceAndMask(...)

## Tuning Random Forest parameter
Please use ./Header/model_selection/tuning_param(...)
