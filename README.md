# Human-In-The-Loop Machine Learning: Comparative Analysis of Active Learning Strategies Using the Example of Visual Coil Inspection![image](https://github.com/user-attachments/assets/fcde7f5e-cc2f-49b8-8b3c-8040a429d423)
![CoverImage](https://github.com/user-attachments/assets/155ce355-9e7f-4b64-9724-3b994be21dd6)
## Aim of the Project
In objective of this study is to evaluate the effectiveness of various Active Learning approaches in reducing labeling effort for defect detection in manufacturing coils. Specifically,this study aims to assess whether AL-based instance selection for labeling can outperform traditional Random Sampling. Additionally, we also compare the effectiveness of various AL query strategies and the performance of the models. We tackle the following research questions -

Research Question 1 : Does active learning surpass random sampling?

Research Question 2: Which active learning query strategies do not result in worse results compared to random sampling?

Research Question 3: Does the gain over random sampling justified considering the overhead costs involved?

We have used DINOv2 and EfficientNetv2 models for the experiments and Optuna for hzperparameter optimisation.

# Project Structure
```
ai-faps-shubham-gupta/
├── DinoLarge/  
│   ├── multiclass/
│   │   ├── ActiveLearning/
│   │   │   ├── MarginSampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_margin_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── margin_sampling.py # Python script for margin sampling
│   │   ├── OptunaStudy/
│   │   │   │   ├── MinicondaName.o967882  # Optuna Study log
│   │   │   │   ├── OptunaStudy02multiclass_DinoL.db  # Optuna db file
│   │   │   │   ├── OptunaStudy02multiclass_DinoL.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multiclass_DinoL.ipynb # Comparision of Active Learning and Randon Sampling
│   ├── multilabel/
│   │   ├── ActiveLearning/
│   │   │   ├── avg_confidence/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_average_confidence.ipynb  # Average result of all the runs
│   │   │   │   ├── average_confidence.py # Python script for avg_confidence
│   │   ├── OptunaStudy/
│   │   │   │   ├── MinicondaName.o968516  # Optuna Study log
│   │   │   │   ├── OptunaStudy01multilabel_DinoL.db  # Optuna db file
│   │   │   │   ├── OptunaStudy01multilabel_DinoL.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multilabel_DinoL.ipynb # Comparision of Active Learning and Randon Sampling
├── DinoSmall/  
│   ├── multiclass/
│   │   ├── ActiveLearning/
│   │   │   ├── EntropySampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_entropy_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── entropy_sampling.py # Python script for entropy sampling
│   │   │   ├── MarginSampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_margin_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── margin_sampling.py # Python script for margin sampling
│   │   │   ├── UncertaintySampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_uncertainty_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── uncertainty_sampling.py # Python script for uncertainty sampling
│   │   ├── OptunaStudy/
│   │   │   │   ├── MinicondaName.o962608  # Optuna Study log
│   │   │   │   ├── optuna_study04multiclass_DInoS.db  # Optuna db file
│   │   │   │   ├── optuna_study04multiclass_DInoS.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multiclass_DinoSmall.ipynb # Comparision of Active Learning and Randon Sampling
│   ├── multilabel/
│   │   ├── ActiveLearning/
│   │   │   ├── avg_confidence/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_average_confidence.ipynb  # Average result of all the runs
│   │   │   │   ├── average_confidence.py # Python script for avg_confidence
│   │   │   ├── avg_score/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_avg_score.ipynb # Average result of all the runs
│   │   │   │   ├── avg_score.py # Python script for avg_score
│   │   │   ├── max_score/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_max_score.ipynb # Average result of all the runs
│   │   │   │   ├── max_score.py # Python script for max_score
│   │   ├── OptunaStudy/
│   │   │   │   ├── MinicondaName.o960242_OptunaStudy01multilabel_DinoS_Layer  # Optuna Study log
│   │   │   │   ├── OptunaStudy01multilabel_DinoS_Layer.db  # Optuna db file
│   │   │   │   ├── OptunaStudy01multilabel_DinoS_Layer.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multilabel_DinoSmall.ipynb # Comparision of Active Learning and Randon Sampling
│   ├── multilabel_samples_from_multiclass/
│   │   ├── ActiveLearning/
│   │   │   ├── EntropySampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_entropy_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── entropy_sampling_multilabel_lesser_lr.py # Python script for entropy sampling
│   │   │   ├── MarginSampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_margin_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── margin_sampling_multilabel_lesser_lr.py # Python script for margin sampling
│   │   │   ├── UncertaintySampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_uncertainty_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── uncertainty_sampling_multilabel_lesser_lr.py # Python script for uncertainty sampling
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling_multiclass_to_multilabel_lesser_lr.py # Python script for random sampling
│   │   ├── compare_multilabel_from_multiclass_DinoSmall.ipynb # Comparision of Active Learning and Randon Sampling
├── EfficientNet/  
│   ├── multiclass/
│   │   ├── ActiveLearning/
│   │   │   ├── EntropySampling/
│   │   │   │   ├── run folder/  # Result 
│   │   │   │   ├── entropy_sampling.py # Python script for entropy sampling
│   │   │   ├── MarginSampling/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_margin_sampling.ipynb  # Average result of all the runs
│   │   │   │   ├── margin_sampling.py # Python script for margin sampling
│   │   │   ├── UncertaintySampling/
│   │   │   │   ├── run folder/  # Result
│   │   │   │   ├── uncertainty_sampling.py # Python script for uncertainty sampling
│   │   ├── OptunaStudy/
│   │   │   │   ├── MinicondaName.o966223  # Optuna Study log
│   │   │   │   ├── OptunaStudy01_multiclass_EffNet.db  # Optuna db file
│   │   │   │   ├── OptunaStudy01multiclass_EffNet.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multiclass_EffNet.ipynb # Comparision of Active Learning and Randon Sampling
│   ├── multilabel/
│   │   ├── ActiveLearning/
│   │   │   ├── avg_confidence/
│   │   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   │   ├── average_average_confidence.ipynb  # Average result of all the runs
│   │   │   │   ├── average_confidence.py # Python script for avg_confidence
│   │   ├── OptunaStudy/
│   │   │   │   ├── OptunaStudy01EffNetMultilabel.log  # Optuna Study log
│   │   │   │   ├── OptunaStudy01_multilabel_EffNet.db  # Optuna db file
│   │   │   │   ├── OptunaStudy01multilabel_EffNet.py # Python script for Optuna Study
│   │   ├── RandomSampling/
│   │   │   ├── run folders/  # Results for different runs based on different seeds
│   │   │   ├── average_random_sampling.ipynb  # Average result of all the runs
│   │   │   ├── random_sampling.py # Python script for random sampling
│   │   ├── compare_multilabel_EffNet.ipynb # Comparision of Active Learning and Randon Sampling
├── CoverImage.jpg # Summary image of the project
├── README.md
├── config.ini # Config file for different experiments
├── requirements.yml
```

## Usage

To clone this repository and navigate into the project directory, run the following commands:

```bash
git clone https://github.com/andi677/ai-faps-shubham-gupta.git
cd ai-faps-shubham-gupta
```

## Create the Environment

To set up the environment, use Conda:

```bash
conda env create -f requirements.yml
conda activate ActiveLearning
```
