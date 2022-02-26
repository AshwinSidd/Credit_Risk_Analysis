# Credit_Risk_Analysis

## Analysis Overview
In this analysis we employed different techniques to train and evaluate models with unbalanced classes using *imbalanced-learn* and *scikit-learn libraries* to build and evaluate models using resampling. For a credit card credit dataset we did oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we used the SMOTEENN algorithm after which we compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Results
### The RandomOversampler Model

![image](https://user-images.githubusercontent.com/92342751/155858694-6966b8d5-fab7-47f9-9660-0228c7a42bc2.png)

From the above image, we can see that the *balanced accuracy score* stood at 63% while the *high risk precision and sensitivity* stood at 1% and 62% respectively. The *low risk precision* values was 100% and *sensitivity* was at 65%.

### SMOTE Oversampling

![image](https://user-images.githubusercontent.com/92342751/155858800-e148a56c-f8ea-43f3-a452-419cbbd64433.png)

From the SMOTE model, we can see that the *balanced accuracy score* stood at 63% while the high risk precision and sensitivity stood at 1% and 62% respectively. The low risk precision values were 100% and sensitivity was at 64%.

### Undersampling using Cluster Centroids

![image](https://user-images.githubusercontent.com/92342751/155858897-682a1bcb-d9da-4490-b7f5-39c44c160eb7.png)

Using CLuster Centroids, we can see that the *balanced accuracy score* stood at 51% while the high risk precision and sensitivity stood at 1% and 59% respectively. The low risk precision values were 100% and sensitivity was at 43%.

### SMOTEENN Model

![image](https://user-images.githubusercontent.com/92342751/155858937-3d8a1860-3628-4948-903b-577726765461.png)

From the SMOTEENN model, we can see that the *balanced accuracy score* stood at 62% while the high risk precision and sensitivity stood at 1% and 71% respectively. The low risk precision values were 100% and sensitivity was at 54%.

### Balanced Random Forest Classifier

![image](https://user-images.githubusercontent.com/92342751/155858977-2fe52e3f-aa6b-46a5-a7a1-3360d1cec7ba.png)

Using Balanced Random Forest Classifier model, we can see that the *balanced accuracy score* stood at 78% while the high risk precision and sensitivity stood at 4% and 67% respectively. The low risk precision values were 100% and sensitivity was at 91%.

### Easy Ensemble AdaBoost Classifier

![image](https://user-images.githubusercontent.com/92342751/155859025-74a91d50-79e6-4750-b29e-32fd0ab2e710.png)

Using Easy Ensemble AdaBoost Classifier model, we can see that the *balanced accuracy score* stood at 92% while the high risk precision and sensitivity stood at 7% and 91% respectively. The low risk precision values were 100% and sensitivity was at 94%.

## Summary
Most models predicted a very low precision for high risk credits. The Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier did show some better results, hence maybe using the Adaboost Classifier model would be better. However, it can also be noted that the dataset given had 99% applications in the low risk category which skews the result immensely. 



