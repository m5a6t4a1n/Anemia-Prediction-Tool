# -*- coding: utf-8 -*-

# The final eight features, in the same order as final_model_feats in training.
CONT_FEATURES = ["Age", "ChronicCount", "BMI", "Sbp", "GripMax"]
CAT_FEATURES = ["Region", "Pension", "ExpectedElderlyCareMode"]
ALL_FEATURES = CONT_FEATURES + CAT_FEATURES

# Continuous variables that must be integers.
INTEGER_FEATURES = ["Age", "ChronicCount"]

# Display names of the features.
FEATURE_LABELS = {
    "Age": "Age (years)",
    "ChronicCount": "Number of chronic diseases",
    "BMI": "BMI (kg/m²)",
    "Sbp": "Systolic blood pressure (mmHg)",
    "GripMax": "Maximum grip strength (kg)",
    "Region": "Geographic region",
    "Pension": "Pension",
    "ExpectedElderlyCareMode": "Expected elderly care mode",
}

# Continuous input control specification: (minimum, maximum, default, step).
# Age and chronic disease count use integer values; the widget therefore shows
# whole numbers without decimal places.
CONT_INPUT_SPEC = {
    "Age":          (50, 100, 65, 1),      # integers; below 60 allowed
    "ChronicCount": (0,  8,   1,  1),      # integers; at most 8
    "BMI":          (12.0, 45.0, 23.0, 0.1),
    "Sbp":          (70.0, 250.0, 130.0, 1.0),
    "GripMax":      (2.0, 80.0, 30.0, 0.5),
}

# Concise hover help for the input controls.
FEATURE_HELP = {
    "Age": "Age in completed years. Enter a whole number. The model is developed mainly for people aged 60 and older; values below 60 are accepted but represent extrapolation.",
    "ChronicCount": "Number of previously diagnosed chronic diseases. Enter a whole number, from 0 to 8.",
    "BMI": "Body mass index = weight (kg) divided by height (m) squared. One decimal place; allowed range 12–45.",
    "Sbp": "Resting systolic blood pressure (the top number), in mmHg, whole number; allowed range 70–250.",
    "GripMax": "Maximum grip strength of the dominant hand, in kg; allowed range 2–80.",
    "Region": "Geographic region of residence, coded into 7 zones; see Feature information in the sidebar for province allocation.",
    "Pension": "Whether the participant currently receives a pension.",
    "ExpectedElderlyCareMode": "Expected primary elderly care arrangement.",
}

# General feature descriptions (used by other scripts if needed).
FEATURE_DESCRIPTIONS = {
    "Age": "Age in years at the time of interview, reflecting the degree of ageing.",
    "ChronicCount": "Total number of previously diagnosed chronic diseases, measuring multimorbidity.",
    "BMI": "Body mass index, weight (kg) divided by height (m) squared, reflecting overall nutrition and body size.",
    "Sbp": "Resting systolic blood pressure (mmHg, the top number), reflecting cardiovascular load.",
    "GripMax": "Maximum grip strength of the dominant hand (kg), reflecting muscle strength and physical function.",
    "Region": "Geographic region of residence, coded into 7 zones.",
    "Pension": "Whether a pension is received, reflecting financial security.",
    "ExpectedElderlyCareMode": "Expected primary elderly care arrangement, reflecting care resource planning.",
}

# Full text for the sidebar "Feature information".
SIDEBAR_FEATURE_DESC = {
    "Age": "Current age in years.",
    "ChronicCount": "Total number of previously diagnosed chronic diseases, including diabetes, "
                    "cancer/malignancy, chronic lung disease, heart disease, stroke, kidney disease, "
                    "stomach/digestive system disease, and arthritis.",
    "BMI": "Body mass index, body weight (kg) divided by the square of height (m). "
           "The normal range for Chinese adults is 18.5–23.9.",
    "Sbp": "Resting systolic blood pressure (the top number), in mmHg. "
           "The normal range is 120–139.",
    "GripMax": "Maximum grip strength of the dominant hand, in kg.",
    "Region": "Geographic region of residence, coded into 7 zones; province allocation is shown below.",
    "Pension": "Whether the participant receives a pension.",
    "ExpectedElderlyCareMode": "Expected primary elderly care arrangement.",
}

# Soft-alert thresholds (retained for reference; the current UI shows no soft alerts).
AGE_EXTRAPOLATION_BELOW = 60
AGE_ELITE_ABOVE = 85
CHRONIC_MULTI_COUNT = 5
GRIP_EXTREME_LOW = 10.0
GRIP_EXTREME_HIGH = 60.0

# BMI bands for Chinese adults: (upper bound, exclusive; alert level; label).
BMI_BANDS = [
    (18.5, "warning", "Underweight (<18.5)"),
    (24.0, "success", "Normal (18.5–23.9)"),
    (28.0, "warning", "Overweight (24–27.9)"),
    (999.0, "warning", "Obese (>=28)"),
]

# Systolic blood pressure bands: (upper bound, exclusive; alert level; label).
SBP_BANDS = [
    (90.0,  "warning", "Possible hypotension (<90)"),
    (120.0, "success", "Normal (90–119)"),
    (140.0, "info",    "High-normal (120–139)"),
    (160.0, "warning", "Grade 1 hypertension (140–159)"),
    (180.0, "warning", "Grade 2 hypertension (160–179)"),
    (999.0, "warning", "Grade 3 hypertension (>=180)"),
]

# Categorical variables: code -> English label (codes identical to the training data).
CAT_LEVELS = {
    "Region": {
        1: "North China",
        2: "Northeast China",
        3: "East China",
        4: "Central China",
        5: "South China",
        6: "Southwest China",
        7: "Northwest China",
    },
    "Pension": {
        0: "No pension",
        1: "Has pension",
    },
    "ExpectedElderlyCareMode": {
        1: "Care by children",
        2: "Savings-based care",
        3: "Pension-based care",
        4: "Other",
    },
}

# Province allocation for the geographic regions (study convention, matching Region codes).
REGION_PROVINCES = {
    1: ("North China", ["Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia"]),
    2: ("Northeast China", ["Liaoning", "Jilin", "Heilongjiang"]),
    3: ("East China", ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Shandong"]),
    4: ("Central China", ["Henan", "Hubei", "Hunan", "Jiangxi"]),
    5: ("South China", ["Guangdong", "Guangxi", "Fujian"]),
    6: ("Southwest China", ["Sichuan", "Chongqing", "Guizhou", "Yunnan"]),
    7: ("Northwest China", ["Shaanxi", "Gansu", "Qinghai", "Xinjiang"]),
}

# Risk stratification thresholds (no longer used by the app; retained for reference).
RISK_THRESHOLDS = {
    "low": 0.10,
    "high": 0.30,
}

# Model and background data file paths.
MODEL_PATH = "anemia_pipeline.joblib"
BACKGROUND_PATH = "background_data.joblib"
