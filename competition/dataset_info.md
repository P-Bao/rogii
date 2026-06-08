# Dataset Info

The competition data comprises horizontal well trajectories and vertical reference logs (Typewells) used for geological prediction. Your goal is to predict the TVT (True Vertical Thickness) for the evaluation zone of each horizontal well.

The data is organized into train/ and test/ directories, where each well is identified by a unique 8-character hash (e.g., 015fe0d2).

## Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| train/ | TBD | TBD | TBD | Contains {WELLNAME}__horizontal_well.csv, {WELLNAME}__typewell.csv, {WELLNAME}.png for each well |
| test/ | TBD | TBD | TBD | Contains horizontal well and typewell files for test wells. TVT target is hidden in evaluation zone. |
| sample_submission.csv | TBD | 2 | TBD | Submission format with id ({WELLNAME}_{row_index}) and tvt |

## Column Descriptions

| Column | Type | Description | Missing % | Notes |
|--------|------|-------------|-----------|-------|
| WELLNAME | string | unique identifier for the well | TBD | |
| MD | float | Measured Depth (ft): The total length of the wellbore from the surface | TBD | |
| X | float | Easting (ft): Spatial coordinate in the horizontal plane | TBD | |
| Y | float | Northing (ft): Spatial coordinate in the horizontal plane | TBD | |
| Z | float | True Vertical Depth (ft): The vertical distance below sea level | TBD | |
| ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA | float | Predicted depth of various geological formations | TBD | Training only |
| TVT | float | True Vertical Thickness (ft): The manually interpreted geological position. Target variable. | TBD | Training only |
| GR | float | Gamma Ray (API): Log measuring natural radioactivity of the rock | TBD | |
| TVT_input | float | Input Target (ft): A copy of TVT provided as a feature | TBD | NaN in evaluation zone |
| Geology | string | Formation Label: categorical label indicating the geological unit | TBD | In typewell file |
| id | string | Unique identifier for prediction point ({WELLNAME}_{row_index}) | 0% | In sample_submission.csv |
| tvt | float | Predicted True Vertical Thickness (ft) | 0% | In sample_submission.csv |

## Class Distribution (Classification)

| Class | Count | Percentage |
|-------|-------|-----------|
| 0 | [FILL] | [FILL]% |
| 1 | [FILL] | [FILL]% |

## Target Distribution (Regression)

- Min: [FILL]
- Max: [FILL]
- Mean: [FILL]
- Std: [FILL]

## Missing Values Summary

[FILL: describe missing patterns, any MCAR/MAR/MNAR patterns observed]

## Key Observations from EDA

- [FILL: observation 1]
- [FILL: observation 2]

## External Data

| Source | Description | Allowed | Notes |
|--------|-------------|---------|-------|
| [FILL] | [FILL] | Yes/No | [FILL] |

## Data Version

Last updated: [FILL date]
