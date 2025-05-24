# Dataset Setup Instructions

## Source:
This project uses a proprietary dataset from the MIT IDSS course:
"Data Science and Machine Learning: Making Data-Driven Decisions," which is why this data has been excluded from this repository.

**Expected structure:**
```
data/
|   tourism_data.csv
```

## Features:
- Gender, Age, City Tier, Occupation
- Duration of Pitch
- Product Pitched, Preferred Property Star
- Number of Followups
- Target: `ProdTaken` (1 = Purchased, 0 = Not purchased)

## Notes:
- Missing values are handled in the script.
- Categorical variables are encoded.
- Train-test split and model tuning are handled in code.