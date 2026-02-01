# --------------------------------------------------
# Step 1: Dataset Review and Modeling Questions
# --------------------------------------------------

"""
Dataset 1: College Completion / Institutional Characteristics

This dataset contains institution-level information about colleges and
universities, including enrollment size, institutional characteristics,
and various financial and outcome-related metrics.

Modeling Question:
Can institutional characteristics and financial investment metrics be
used to predict student enrollment size at a college or university?

Target Variable:
student_count

I chose student_count as the target because it is a clear, measurable
outcome that reflects institutional scale and is likely influenced by
many of the variables included in the dataset.
"""

"""
Dataset 2: Job Placement

This dataset contains student-level academic, demographic, and
professional information, along with whether or not each student
was successfully placed in a job.

Modeling Question:
Can academic performance, educational background, and work experience
be used to predict whether a student is successfully placed in a job?

Target Variable:
status (Placed vs. Not Placed)

This target directly represents the placement outcome, which makes it
appropriate for a classification problem.
"""

# --------------------------------------------------
# Step 2: Business Metric and Data Preparation
# --------------------------------------------------

"""
Dataset 1: College Completion

Business Metric:
The primary metric of interest for this dataset is student enrollment
size. Understanding which institutional and financial factors are
associated with enrollment can help inform decisions about funding,
resource allocation, and institutional planning.

Data Preparation Decisions:
- Institutional characteristics such as control, level, and classification
  are treated as categorical variables.
- Financial, enrollment, and rate-based variables are treated as continuous.
- Identifier and location variables that do not directly contribute to
  predicting enrollment are removed.
- Low-frequency categorical levels are collapsed to reduce sparsity.
- Categorical variables are one-hot encoded.
- Continuous variables are normalized.
- student_count is defined as the target variable.
- The data is split into training, tuning, and testing sets.
"""

"""
Dataset 2: Job Placement

Business Metric:
The primary metric for this dataset is job placement rate. Being able to
predict placement outcomes can help institutions identify which factors
are most important for employability and which students may need
additional support.

Data Preparation Decisions:
- Demographic and academic background variables (e.g., gender, degree type,
  specialization, work experience) are treated as categorical.
- Academic percentage variables are treated as numeric values on a 0–100 scale.
- The salary variable is dropped because it is only observed after placement
  and would introduce target leakage.
- Low-frequency categorical levels are collapsed when necessary.
- Categorical variables are one-hot encoded.
- Continuous variables are normalized.
- status is defined as the target variable.
- Class prevalence is calculated to understand class balance.
- The data is split into training, tuning, and testing sets.
"""

# --------------------------------------------------
# Step 3: Data Instincts and Potential Concerns
# --------------------------------------------------

"""
Dataset 1: College Completion

This dataset appears capable of addressing the modeling question since it
includes a broad range of institutional and financial variables that are
likely related to enrollment size. However, some financial and rate-based
variables may be missing or inconsistently reported across institutions,
which could reduce the amount of usable data. Additionally, enrollment size
is influenced by external factors such as regional population trends or
institutional reputation that are not captured in the dataset, which may
limit predictive performance.
"""

"""
Dataset 2: Job Placement

This dataset is well-suited for predicting placement outcomes because it
contains academic performance, educational background, and work experience
variables that are directly related to employability. One concern is the
possibility of class imbalance if most students are placed. In addition,
several academic percentage variables may be highly correlated, which could
affect model stability. There are also potential fairness and generalizability
concerns, since some variables may indirectly reflect background or
opportunity differences.
"""
