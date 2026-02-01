# ml_week_3_project
## Step 1: Dataset Review and Modeling Questions

### Dataset 1: College Completion / Institutional Characteristics

**Dataset Overview**  
This dataset contains institution-level information for U.S. colleges and universities, including location, institutional control, level (2-year vs. 4-year), classification, enrollment size, and multiple financial award and expenditure metrics.

**Modeling Question**  
Can institutional characteristics and financial investment metrics be used to predict student enrollment size at a college or university?

**Target Variable**  
- `student_count` (continuous; regression problem)
- 

### Dataset 2: Job Placement

**Dataset Overview**  
This dataset contains individual-level academic, demographic, and professional attributes of students, including academic performance, educational background, work experience, and job placement outcomes.

**Modeling Question**  
Can academic performance, educational background, and work experience be used to predict whether a student is successfully placed in a job?

**Target Variable**  
- `status` (binary categorical; classification problem: Placed vs. Not Placed)


## Step 2: Business Metric and Data Preparation

### Dataset 1: College Completion / Institutional Characteristics

**Business Metric**  
The primary business metric for this problem is student enrollment size. Understanding how institutional characteristics and financial investment metrics relate to enrollment can help institutions and policymakers allocate resources more effectively and support sustainable enrollment outcomes.

**Data Preparation Steps**
1. Correct variable types by treating institutional characteristics (e.g., control, level, classification, HBCU status) as categorical variables and financial or enrollment-related measures as continuous numeric variables.
2. Interpret percentage- and rate-based variables as institution-level metrics rather than individual student scores.
3. Drop unneeded identifier and location variables that do not directly contribute to modeling enrollment size.
4. Collapse low-frequency categorical levels where necessary to reduce sparsity.
5. One-hot encode categorical variables such as institutional control and institution level.
6. Normalize continuous variables including financial award amounts, expenditure metrics, and rate-based measures.
7. Define the target variable as `student_count`.
8. Calculate summary statistics of the target variable to understand its distribution.
9. Split the data into training, tuning, and testing partitions to support model development and evaluation.

---

### Dataset 2: Job Placement

**Business Metric**  
The primary business metric for this problem is job placement rate. Accurately predicting placement outcomes can help institutions identify factors that contribute to successful placements and provide targeted support to students who may be at risk of not being placed.

**Data Preparation Steps**
1. Correct variable types by treating demographic, academic background, and specialization variables (e.g., gender, degree type, specialization, work experience) as categorical variables and academic performance measures as continuous numeric variables.
2. Interpret percentage-based variables as raw academic or test score percentages on a 0–100 scale rather than percentiles.
3. Drop unneeded identifier variables such as serial number fields.
4. Collapse rare categorical levels where appropriate to reduce overfitting.
5. One-hot encode categorical variables including degree type, specialization, and work experience.
6. Normalize continuous variables such as academic performance and test score percentages.
7. Define the target variable as `status` (Placed vs. Not Placed).
8. Calculate the prevalence of the target variable to assess class balance.
9. Split the data into training, tuning, and testing partitions to support classification modeling.

## Step 3: Data Instincts and Potential Concerns

### Dataset 1: College Completion / Institutional Characteristics

This dataset seems appropriate for the question because it includes a variety of institutional characteristics and financial metrics that could reasonably be related to enrollment size. That said, there are a few things to be cautious about. Some of the financial and rate-based variables may have missing or inconsistent values across institutions, which could reduce the amount of usable data after cleaning. In addition, certain categorical variables may have many low-frequency categories, which could make the data more sparse once encoded. Finally, enrollment size is likely influenced by factors not captured in the dataset, such as regional population trends or institutional reputation, which may limit how well the model can explain or predict enrollment.

### Dataset 2: Job Placement

This dataset is well-suited for predicting job placement since it includes academic performance, educational background, and work experience variables that are directly related to employability. One possible concern is class imbalance if most students in the dataset are placed, which could cause a model to favor the majority class. Some of the academic percentage variables may also be highly correlated with one another, which could affect model stability. Additionally, certain variables may indirectly reflect background or opportunity differences, which could limit how well the model generalizes to other groups or settings.
