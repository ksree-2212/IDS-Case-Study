# Talent Lifecycle Analytics: Candidate Stability and Employee Attrition Prediction Using R

## Introduction

**Talent Lifecycle Analytics** is an end-to-end Introduction to Data Science Lab case study that uses R and machine learning to examine two stages of the employee lifecycle: **candidate job-change intent** and **employee attrition**. It analyses candidate profiles to estimate whether a candidate is *not currently seeking a job change* and analyses employee records to estimate which workplace profiles are associated with leaving. The notebook combines data cleaning, exploratory visualisation, logistic regression, model evaluation, LIME explanations, association-rule mining, candidate shortlisting, and an exploratory comparison by experience band.

> **Important interpretation note:** The candidate dataset does not contain actual offer acceptance or joining outcomes. In this project, `Likely_Join = Yes` is a proxy meaning that the source dataset labels the candidate as *not currently seeking a job change*. It is not a verified probability that the candidate will accept an offer or join a particular employer. The employee dataset is a simulated IBM HR benchmark dataset, not actual company employee records.

## Project at a glance

| Item | Details |
|---|---|
| Project title | Talent Lifecycle Analytics: Predicting Candidate Joining Likelihood and Employee Attrition Risk Using R |
| Course | 22ADC54N - Introduction to Data Science Lab |
| Language | R |
| Main model | Logistic Regression (`glm`, binomial family) |
| Model evaluation | Stratified 75/25 train-test split; accuracy, precision, recall, specificity, ROC curve, and AUC |
| Explainability | LIME (Local Interpretable Model-agnostic Explanations) |
| Pattern discovery | Apriori association rules using `arules` |
| Visualisation | `ggplot2`, `scales`, and `corrplot` |

## Problem statement

HR teams need ways to understand candidate job-change intent and employee attrition risk. These are related workforce-planning questions, but the available datasets do not follow the same people from application through employment and exit. This project therefore analyses the two datasets separately and compares their group-level results by experience band as an **exploratory indicator**, not as an end-to-end prediction for the same individuals.

## Objectives

1. Inspect and prepare public candidate and employee HR datasets.
2. Explore missing values, class balance, and patterns in education, experience, company information, job role, satisfaction, overtime, income, and other HR attributes.
3. Build logistic regression models for the candidate stability proxy and employee attrition.
4. Evaluate each model on held-out test data and compare a default probability threshold of 0.50 with a threshold selected from training predictions using Youden's J statistic.
5. Explain individual predictions with LIME.
6. Discover combinations of attributes associated with outcomes using Apriori association rules.
7. Rank candidate test records into five probability tiers and compare predicted proxy rates with observed rates.
8. Compare candidate stability-proxy rates with employee stay rates across experience bands.

## Datasets

### 1. Candidate profiles - `aug_train.csv`

- **Records:** 19,158
- **Original columns:** 14
- **Source:** Public HR Analytics / Job Change of Data Scientists dataset (Kaggle-format `aug_train.csv`)
- **Target used in this notebook:** `target`
- **Project label:** `Likely_Join`

Selected features include city development index, gender, relevant work experience, university enrolment, education level, major discipline, years of experience, company size, company type, time since last job change, and training hours.

The project maps the source target to `Likely_Join` as follows:

| Project label | Meaning in this dataset |
|---|---|
| `Likely_Join = Yes` | Source target is 0: candidate is not currently seeking a job change |
| `Likely_Join = No` | Candidate is currently seeking a job change |

This mapping is a proxy for candidate stability/job-change intent, **not actual joining behaviour**.

### 2. Employee attrition - IBM HR Analytics

- **Records:** 1,470
- **Original columns:** 35
- **Source:** IBM HR Analytics Employee Attrition benchmark dataset (simulated)
- **Target:** `Attrition`

The dataset contains employee demographics, job information, work environment measures, satisfaction ratings, income, overtime, tenure, and other HR attributes. The notebook removes constant columns and the employee identifier before modelling. The overall `Attrition = Yes` rate in the notebook output is approximately **16.1%**.

**Data links:**
- Candidate dataset listing: [HR Analytics - Job Change of Data Scientists on Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- Employee dataset source used by the notebook: [IBM HR Attrition CSV on GitHub](https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/employee_attrition/HR-Employee-Attrition-All.csv)

## Methodology and workflow

### Step 1 - Load and inspect data

The notebook loads `aug_train.csv` from the working directory and downloads the IBM HR attrition CSV, with a local-file fallback for the employee data. It checks dimensions, column structure, missing values, and duplicate records.

### Step 2 - Clean and prepare the data

**Candidate module**

- Converts experience values such as `<1` and `>20` into numeric approximations.
- Converts `last_new_job` values such as `never` and `>4` to numeric categories.
- Fills missing numeric values with the median and replaces missing categorical values with `Unknown`.
- Derives relevant-experience and experience-band fields.
- Uses the project target `Likely_Join` for modelling.

The candidate dataset has particularly high missingness in employer and demographic fields: `company_type` (about 32.0%), `company_size` (31.0%), `gender` (23.5%), and `major_discipline` (14.7%).

**Employee module**

- Drops constant fields such as `EmployeeCount`, `Over18`, and `StandardHours`, plus the employee ID.
- Converts categorical fields to factors and uses the same experience bands for comparison.
- The source employee dataset has 1,470 rows and no missing values in the notebook output.

### Step 3 - Exploratory Data Analysis (EDA)

The notebook uses bar charts, density plots, boxplots, stacked bars, a correlation heatmap, and an education-by-experience heatmap to explore distributions and relationships before modelling.

### Step 4 - Logistic regression

Each module uses logistic regression with a binary outcome. The data is split into **75% training and 25% testing**, preserving the target class proportions. The candidate model uses 14,369 training records and 4,789 test records; the employee model uses 1,103 training records and 367 test records.

### Step 5 - Threshold selection and evaluation

The notebook evaluates predictions at both `0.50` and a tuned threshold selected using **training-set predictions only** with Youden's J statistic. Performance is measured with accuracy, recall, specificity, precision, the confusion matrix, and ROC-AUC. AUC does not change when only the classification threshold changes.

### Step 6 - Explain individual predictions with LIME

LIME gives local, approximate explanations for selected cases: six high/low-probability candidate cases and four high-risk employee cases. The blue and red bars indicate features that support or contradict the model's prediction for that individual; they are not causal explanations.

### Step 7 - Mine association rules

Apriori identifies combinations of attributes that co-occur with `Likely_Join = Yes`, `Likely_Join = No`, or `Attrition = Yes`. Results are ranked by lift and include support and confidence. The notebook also examines rules without `Unknown` categories, since `Unknown` means the field was missing rather than a real employer category.

### Step 8 - Candidate shortlist and lifecycle comparison

Held-out candidate records are ranked by predicted proxy probability and divided into five tiers. The notebook compares mean predicted probability with the observed proxy rate within each tier. Finally, candidate proxy rates and employee stay rates are compared by experience band. The two datasets represent different people, so this comparison is descriptive and exploratory only.

## Results from the notebook

### Model performance

The values below are from the included notebook PDF output. `Yes` is the positive class in each model: `Likely_Join = Yes` means **not currently seeking a job change**; `Attrition = Yes` means the employee **left**.

| Model | Threshold | Accuracy | Recall | Specificity | Precision | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Candidate stability proxy | 0.50 | 0.767 | 0.932 | 0.271 | 0.794 | 0.770 |
| Candidate stability proxy | **0.76 (tuned)** | 0.726 | 0.721 | 0.744 | 0.894 | 0.770 |
| Employee attrition | 0.50 | 0.850 | 0.203 | 0.974 | 0.600 | 0.768 |
| Employee attrition | **0.26 (tuned)** | 0.837 | 0.559 | 0.890 | 0.493 | 0.768 |

**Interpretation:** On the candidate model, tuning the threshold reduces the tendency to classify almost everyone as `Yes` and improves the balance between recall and specificity. On the attrition model, the tuned threshold increases recall from about 20.3% to 55.9%, helping identify more employees who left, while reducing precision. Threshold choice depends on the cost of missed cases versus false alarms.

### Candidate class balance and missing values

The candidate dataset has about **75.1% `Likely_Join = Yes`** and **24.9% `Likely_Join = No`** under the project's proxy definition. Missing employer information is common, so those missing fields are retained as an explicit `Unknown` category rather than dropping a large portion of the records.

![Candidate target distribution from notebook output](images/candidate_target_distribution.png)

### EDA: candidate experience and education

The output charts show that the not-seeking-job-change proxy rate varies by education and experience band; in the notebook output, that rate rises from approximately **61.6%** for 0-2 years of experience to **84.4%** for 16+ years. This is a group-level association, not evidence that experience causes a candidate to join.

![Candidate education and experience charts from notebook output](images/candidate_eda_experience_education.png)

### LIME explanations for candidate predictions

The six local explanations show which features push individual candidate predictions toward or away from the not-seeking-job-change proxy class.

![Candidate LIME explanations from notebook output](images/candidate_lime_explanations.png)

### Candidate shortlist tiers

The held-out candidate records are divided into five approximately equal-sized tiers based on predicted probability. In the notebook run, the summary was:

| Tier | Number of candidates | Mean predicted proxy probability | Observed proxy rate |
|---:|---:|---:|---:|
| 1 (highest predicted) | 958 | 93.5% | 91.4% |
| 2 | 958 | 89.6% | 91.4% |
| 3 | 958 | 83.3% | 85.7% |
| 4 | 958 | 67.3% | 61.8% |
| 5 (lowest predicted) | 957 | 43.2% | 44.9% |

![Candidate shortlist tier chart from notebook output](images/candidate_shortlist_tiers.png)

### Employee attrition patterns

The employee dataset has **1,470 records** and an overall attrition rate of **16.1%**. The exploratory charts examine attrition across job roles and experience bands; the highest attrition rates in the notebook appear in early-career groups and selected roles, which can help motivate further investigation by HR.

![Employee attrition rate by job role from notebook output](images/attrition_by_job_role.png)

![Employee attrition rate by total work experience from notebook output](images/attrition_by_experience.png)

### Employee model ROC curve and LIME

The employee attrition model has a test-set AUC of **0.768**. At the tuned threshold of **0.26**, its test-set recall is approximately **55.9%** and specificity is approximately **89.0%**.

![Employee attrition ROC curve from notebook output](images/attrition_roc_curve.png)

LIME is also used to explain four high-risk employee records, showing local feature contributions such as overtime, total working experience, satisfaction, and other inputs used by the model.

![Employee LIME explanations from notebook output](images/employee_lime_explanations.png)

### Talent lifecycle comparison by experience band

| Experience band | Candidate `Likely_Join = Yes` proxy rate | Employee stay rate (`Attrition = No`) |
|---|---:|---:|
| 0-2 years | 61.6% | 56.1% |
| 3-5 years | 67.8% | 80.8% |
| 6-10 years | 74.7% | 85.0% |
| 11-15 years | 80.9% | 87.4% |
| 16+ years | 84.4% | 91.3% |

![Candidate stability proxy and employee stay rates by experience band](images/talent_lifecycle_comparison.png)

These figures compare groups from two independent datasets. They do **not** show what happened to the same candidate after being hired and should not be interpreted as causal or individual-level lifecycle probabilities.

## Key libraries

- **`tidyverse`** - data cleaning, transformation, summaries, and `ggplot2` charts
- **`caret`** - stratified train/test partitions and cross-validated fitting for LIME explanations
- **`lime`** - local explanations for individual predictions
- **`arules`** - Apriori association-rule mining
- **`scales`** - percentage and numeric chart formatting
- **`corrplot`** - correlation heatmap

## How to run the project

1. Download or clone this repository.
2. Open `Talent_Lifecycle_Case_Study.ipynb` in a Jupyter environment with an **R kernel/runtime** (or an equivalent Colab R setup).
3. Make sure `aug_train.csv` is in the notebook's working directory / runtime Files area.
4. Run the notebook **from the first cell to the last cell**. The first run may take several minutes while R packages are installed.
5. Keep internet access enabled for package installation and for the IBM HR dataset download, unless you provide the local fallback CSV expected by the notebook.
6. Check the generated metrics and output charts in your own run before quoting results in a presentation.

### Files generated by the notebook

| File | Purpose |
|---|---|
| `top_candidates_shortlist.csv` | Highest-ranked candidate records in the held-out test subset, ranked by the proxy probability |
| `candidate_join_rules.csv` | Apriori rules associated with `Likely_Join = Yes` |
| `candidate_notjoin_rules.csv` | Apriori rules associated with `Likely_Join = No` |
| `attrition_association_rules.csv` | Apriori rules associated with `Attrition = Yes` |
| `model_metrics.csv` | Performance metrics for the default and tuned thresholds for both models |
| `hiring_lime_explanations.png` | High-resolution candidate LIME output generated by the notebook |

## Limitations and responsible use

- **Proxy target:** The candidate target is job-change intent, not an observed hiring outcome.
- **Separate datasets:** Candidates and employees are not linked at an individual level.
- **Simulated employee dataset:** The IBM HR attrition data is a benchmark dataset and may not reflect a specific organisation.
- **Class imbalance:** Only about 16.1% of employee records have `Attrition = Yes`; accuracy alone is not enough to assess the retention model.
- **Association is not causation:** Logistic regression coefficients, LIME explanations, and Apriori rules show model associations or local approximations, not proof that a factor causes an outcome.
- **Validation improvement:** The current notebook prepares/imputes some candidate fields before the train/test split. A more rigorous production pipeline should estimate imputation and other preprocessing parameters on the training data only, then apply them to held-out data.
- **Fairness and human review:** HR predictions can affect people's opportunities. Before any real-world use, evaluate bias, data quality, calibration, and legal/privacy requirements; treat model output as decision support, never as an automatic hiring or termination decision.

## Future scope

- Use linked organisational data that records application, offer, joining, and exit outcomes.
- Add useful validated features such as interview feedback, skills assessment, job level, and role requirements.
- Compare logistic regression with other models using the same leakage-safe train/test design and suitable validation.
- Add fairness checks, probability calibration, temporal validation, and clearer monitoring of false positives and false negatives.
- Build an interactive HR analytics dashboard for candidate shortlisting and retention-risk monitoring.

## Repository structure

```text
Talent-Lifecycle-Analytics/
├── README.md
├── Talent_Lifecycle_Case_Study.ipynb
├── aug_train.csv
└── images/
    ├── candidate_target_distribution.png
    ├── candidate_eda_experience_education.png
    ├── candidate_lime_explanations.png
    ├── candidate_shortlist_tiers.png
    ├── attrition_by_job_role.png
    ├── attrition_by_experience.png
    ├── attrition_roc_curve.png
    ├── employee_lime_explanations.png
    └── talent_lifecycle_comparison.png
```

## Conclusion

This case study demonstrates a repeatable R-based workflow for inspecting HR datasets, modelling two binary outcomes, evaluating predictions, explaining individual predictions, discovering recurring patterns, and summarising findings by experience band. Its main value is to show how interpretable analytics can support better questions and more focused HR review, while being explicit about what the datasets can—and cannot—tell us.
