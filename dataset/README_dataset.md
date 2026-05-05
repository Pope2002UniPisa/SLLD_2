# Prepared SLLD Dataset

Dataset extracted from `SLLD_exam.zip`, folder `word_ratings`, using `Word_Ratings.csv` as the main data source.

## Source

The dataset is based on Binder et al. (2016), *Toward a brain-based componential semantic representation*. The source readme states that the spreadsheet contains conceptual attribute ratings and lexical variables for English words.

## Prepared files

- `slld_binder_base_65_features.csv`: main cleaned dataset with metadata, target labels, 65 semantic features and category labels.
- `slld_binder_Xy_65_features.csv`: compact modelling dataset with `entry_id`, `word`, `target_word_class` and the 65 semantic features.
- `slld_binder_Xy_2210_expanded_terms.csv`: high-dimensional modelling dataset with 65 original features, 65 squared terms and all pairwise interactions: `65 + 65 + C(65, 2) = 2210` predictors.
- `slld_binder_feature_dictionary.csv`: feature dictionary with domain and description of the 65 conceptual attributes.
- `slld_binder_dataset_summary.json`: machine-readable summary of dimensions, target and missing values.

## Dimensions

- Lexical entries: 535
- Unique word forms: 534
- Base semantic features: 65
- Expanded terms: 2210

## Target

The supervised target is `target_word_class`, derived from the original `WC` column:

- `1` = noun
- `2` = verb
- `3` = adjective

Class distribution:

- noun: 434
- verb: 62
- adjective: 39

## Missing values

Missing conceptual ratings are kept as empty cells. They are not imputed in the prepared files.

Missing base features:

- `complexity`: 101
- `practice`: 39
- `caused`: 39
- `drive`: 1

## Important preprocessing note

No imputation, standardization or train/test split has been applied here. This is intentional: missing-value imputation and scaling should be fitted on the training set only, then applied to validation/test data, in order to avoid data leakage.

The only duplicate word form is `used`, which appears twice in the original dataset because it was rated separately as a verb and as an adjective. Both rows are retained because they correspond to different lexical entries and different target classes.
