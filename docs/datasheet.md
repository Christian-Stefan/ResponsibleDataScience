# Dataset Datasheet

## Motivation

1. **Purpose**: This dataset was created to develop and validate ICU mortality prediction models. It contains patient data from intensive care units, including vital signs, laboratory results, and demographic information.

2. **Creators**: The data was collected through MIT's GOSSIS (Global Open Source Severity of Illness Score) initiative.

3. **Funding**: [To be completed with funding information]

## Composition

1. **Data Instances**: The dataset contains over 130,000 ICU visits recorded over one year.

2. **Fields/Features**:
   - Demographic information (age, gender, ethnicity)
   - Vital signs (heart rate, blood pressure, temperature, etc.)
   - Laboratory results (blood tests, etc.)
   - Apache scores and predictions
   - Target variable: hospital mortality

3. **Missing Data**: 
   - Detailed analysis of missing values per feature
   - Patterns in missing data
   - Handling strategies implemented

4. **Confidentiality**: All data has been de-identified following HIPAA guidelines.

## Collection Process

1. **Acquisition**: Data was collected from ICU electronic health records.

2. **Sampling**: [To be completed with sampling strategy details]

3. **Time Frame**: One year of ICU visits.

4. **Data Validation**: [To be completed with validation process details]

## Preprocessing/Cleaning

1. **Raw Data**: Original data contains [X] features and [Y] instances.

2. **Preprocessing Steps**:
   - Removal of features with high missing values (>80%)
   - Imputation strategies for remaining missing values
   - Feature encoding and scaling
   - Handling of outliers

3. **Data Quality Measures**: [To be completed with quality assurance steps]

## Uses

1. **Intended Uses**: 
   - Development of ICU mortality prediction models
   - Comparison with existing APACHE scoring systems
   - Research on healthcare disparities and fairness

2. **Known Misuses**: 
   - Should not be used as the sole basis for clinical decisions
   - Not suitable for real-time predictions without proper validation

3. **Ethical Considerations**:
   - Potential biases in data collection
   - Impact on different demographic groups
   - Privacy considerations

## Distribution

1. **Access**: Data is available through PhysioNet with appropriate credentials and data use agreement.

2. **License**: [To be completed with license information]

## Maintenance

1. **Updates**: [To be completed with update schedule/policy]

2. **Versions**: [To be completed with version information]

3. **Contact**: [To be completed with contact information]

## Technical Specifications

1. **File Formats**: CSV files

2. **Storage Requirements**: Approximately [X] GB

3. **Software Requirements**: 
   - Python 3.10+
   - Required packages listed in requirements.txt 