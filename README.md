# Speaker Age Prediction from Audio and Metadata

This project presents a supervised machine Learning approach for predicting speaker age from structured speech-related features and metadata. The task is formulated as a regression problem, where the target variable is speaker age and the input features include extracted audio descriptors, pitch and voice-quality measures, silence and pause features, speech-duration indicators, and metadata.

The project compares a naive baseline model with more advanced tree-based regression models. The final objective is not only to obtain accurate predictions, but also to understand which feature groups contribute most to age prediction and where the model performs less reliably.

## Project Overview

Automatic age prediction from speech is based on the idea that speech patterns contain age-related information. In this project, different types of speech and metadata features are used to estimate speaker age.

The workflow includes:

- Data preprocessing and feature preparation
- Exploratory data analysis
- Baseline model construction
- Model comparison using cross-validation
- Hyperparameter tuning
- Final test-set evaluation
- Feature-group comparison
- Permutation feature importance
- Residual and prediction-error analysis

## Dataset

The dataset contains structured speech-related data with:

- 2,933 samples
- 20 original columns
- Speaker ages ranging from 6 to 97 years
- A right-skewed age distribution, with more younger speakers than older speakers

This imbalance is important because the final model performs more reliably for younger speakers, while predictions for older speakers are less stable due to fewer examples in that age range.

## Models

Three regression models were evaluated:

| Model | Purpose |
|---|---|
| DummyRegressor | Baseline model using the mean age |
| ExtraTreesRegressor | Tree-based ensemble model for nonlinear patterns |
| HistGradientBoostingRegressor | Gradient-boosted tree model selected as the final model |

The DummyRegressor provides a naive reference point. ExtraTrees and HistGradientBoosting were used because tree-based ensemble models can capture nonlinear relationships and feature interactions in tabular data.

## Evaluation

The main evaluation metric is RMSE, reported in years. RMSE is suitable because age is a continuous target variable and the error can be interpreted directly as the average prediction error scale.

Cross-validation was used to compare models more reliably and avoid depending on a single train-test split.

## Results

The best-performing model was the tuned HistGradientBoostingRegressor.

| Metric | Result |
|---|---|
| Baseline test RMSE | approximately 12.92 years |
| Tuned HistGradientBoosting test RMSE | approximately 9.04 years |
| Tuned HistGradientBoosting test MSE | approximately 81.69 |
| Best cross-validation RMSE after tuning | approximately 10.00 years |

The tuned HistGradientBoosting model reduced the test RMSE by nearly four years compared with the baseline. This shows that the speech-related features and metadata contain meaningful predictive signal for speaker age estimation.

## Feature Analysis

Feature-group experiments showed that the best performance was achieved when all feature types were combined. Extracted audio features alone also performed strongly, while pitch and voice-quality features alone were less effective.

The results suggest that age-related information is distributed across multiple aspects of speech, including:

- Audio duration
- Silence and pause behavior
- Spectral features
- Pitch-related features
- Voice-quality measures
- Metadata

Permutation importance showed that the model does not depend on one single feature. Instead, it combines several feature types to improve prediction accuracy.

## Prediction Diagnostics

The final model shows good overall performance, but the diagnostic plots reveal uneven reliability across the age range. Predictions are more accurate for younger speakers, where the dataset has more examples. For older speakers, the model tends to pull predictions closer to the middle age range, which is a common regression-to-the-mean effect.

The residual distribution is centered close to zero, suggesting that the model does not have a strong general tendency to overpredict or underpredict. However, the presence of larger errors in the tails shows that some individual predictions remain difficult.

## Limitations

The main limitation of the project is the imbalance in the age distribution. Since younger speakers are much more frequent than older speakers, the model learns younger age patterns more effectively.

Other limitations include:

- RMSE may hide differences in performance across age groups
- Some metadata variables may introduce fairness or bias concerns
- Feature importance indicates predictive association, not causation
- Older-speaker prediction reliability should be studied more carefully


## Conclusion

This project demonstrates that speaker age can be predicted meaningfully from structured audio and metadata features. Among the tested models, HistGradientBoosting achieved the strongest performance after hyperparameter tuning. The best results were obtained by combining audio, temporal, voice-quality, and metadata features rather than relying on a single feature group.

Although the results are promising, the model should be interpreted carefully because prediction reliability varies across the age distribution, especially for older speakers.


**Author:** Fatemeh Saleh  
**Institution:** Politecnico di Torino  
**Student ID:** s344410
