# Application Guides

## Data Preprocessing

<p align="center">
  <img src="assets\data_preprocessing.gif" />
</p>

- Get the raw dataset from selecting file from the folder ``datasets/raw_dataset/train_dataset`` or uploading a file from the local computer.
- The folder contains several CSV files as experiment examples.  
  - Classification &rarr; id: id, target: loan_status (Loan Approval Prediction)
  - Multi-classification &rarr; id: id, target: Target (Students' Dropout and Academic Success Prediction)
  - Regression &rarr; id: id, target: price (Gemstone Price Prediction)
- After setting configuration for data input, click ``Input Data`` button and then the screen will show raw dataset and data quality visualization if enabled.
> [!WARNING]
> For multi-classification, make sure that the target column should be just one column from the dataset with categorical type.
- Set configurations on how dataset will be preprocessed and define the data preprocessing name to save the cleaned dataset.
- Click ``Process Data`` button and then the dataset is preprocessed, saved, and showed in the screen. The cleaned dataset is saved in the ``datasets/cleaned_dataset/[DP_NAME]`` folder as ``cleaned_train.parquet``.

## Exploratory Data Analysis

| Univariate Analysis    | Bivariate Analysis     |
| :--------------------: | :--------------------: |
| ![](assets\eda_ua.png) | ![](assets\eda_ba.png) |

Select the cleaned dataset that will be explored (both for univariate and bivariate analysis) then set configurations based on the visualization.

## Model Training

<p align="center">
  <img src="assets\training.gif"/>
</p>

- Set configurations for the experiment with the experiment name. Choose feature engineering, models, and metrics that will be used.
> [!WARNING]
> Make sure that the target column in the dataset is suitable with ML task chosen. If not considered, then the error will be happen.  
> 
> Example: It will apply classification, but the target column use numerical type with many unique values that should be applied for regression. Consequently, the application is not able to train the model.
>
> Furthermore, if using categorical encoding, ML models other than advanced ensemble-tree based (XGBoost, LightGBM, and CatBoost) can only use `One-Hot` encoding type. 
- Click ``Train the Model`` button. Then, models will be trained. Configs, models, logs, and results will be saved in the ``experiments/[EXP_NAME]`` folder.

## Model Evaluation

<p align="center">
  <img src="assets\evaluation.png" width="720"/>
</p>

Select the experiment results that will be evaluated then set configurations based on the visualization.

## Model Explainability

<p align="center">
  <img src="assets\explainability.gif"/>
</p>

- Define the experiment name and model that will be explained.
> [!NOTE]
> Model explainability can only be used for tree-based models.
- Click ``Explain the Model`` button and the screen will show global and local explanations. The results are showed based on valid dataset.
- To effectively use local explanations to know how the model behave and predict the label for specific data, check the id column from ``Easy-Difficult Samples`` section. Choose the ``index`` column value that corresponds with the id column's data. Insert into the local explanation field, then the explanation for this data will be showed.

## Model Prediction

<p align="center">
  <img src="assets\prediction.gif"/>
</p>

- Get the raw test dataset from selecting file from the folder ``datasets/raw_dataset/test_dataset`` or uploading a file from the local computer.
- Same as data preprocessing, the folder contains several CSV files as testing examples. The difference is there are labeled and unlabeled raw test dataset.
- After setting the file and the data preprocessing name, click ``Input Data`` button and the test dataset will be cleaned and saved in the ``datasets/cleaned_dataset/[DP_NAME]`` folder as ``cleaned_test.parquet``.
> [!WARNING]
> The train and test dataset structure must be same based on the data preprocessing name to avoid unnecessary errors.
- Data comparison between train and test dataset will also be showed if enabled.
- Set testing configurations with the test name. After that, click ``Test the Model`` button. The results will be saved in the ``experiments/[EXP_NAME]/tests/[TEST_NAME]`` folder.
- Testing results is showed in the screen with local explanations. If the test dataset have labels, metrics also be showed.