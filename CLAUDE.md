# Tax-Microdata-Benchmarking (TMD): Comprehensive Technical Documentation

## Project Overview

The **Tax-Microdata-Benchmarking (TMD)** project creates synthetic tax microdata files for federal tax policy analysis, serving as the data foundation for two major tax microsimulation models: **PolicyEngine-US** and **Tax-Calculator**. The project builds high-quality synthetic datasets that combine survey data's statistical representativeness with administrative tax data's detail, enabling accurate tax revenue modeling and policy analysis.

## Core Data Sources and Integration

### Primary Data Sources

1.  **2015 IRS Statistics of Income Public Use File (PUF)** - 207,697 detailed actual tax return records
2.  **Current Population Survey (CPS) 2021/2022** - Population demographics and economics covering both filers and non-filers
3.  **IRS Statistics of Income published tables** - Validation targets and benchmarking data
4.  **Federal agency data** (CBO, Treasury, JCT) - Growth factors, revenue estimates, and tax expenditure targets

### Key Output Files

-   **`tmd.csv.gz`**: Main microdata file with \~230,000 records containing 100+ tax variables
-   **`tmd_weights.csv.gz`**: Population weights for different years and geographic areas
-   **`tmd_growfactors.csv`**: Economic growth factors for temporal extrapolation
-   Comprehensive validation reports comparing model outputs to official estimates

## CPS Data Processing: Technical Deep-Dive

### CPS Data Sources and Loading

The project uses Current Population Survey Annual Social and Economic Supplement (ASEC) data from the U.S. Census Bureau:

-   **Years Available**: 2018-2022 (survey years), corresponding to tax years 2017-2021
-   **Primary Focus**: 2021 CPS data (administered in March 2022)
-   **Data Format**: ZIP files containing CSV files from Census Bureau

**CPS URLs by Year** (from `tmd/datasets/cps.py`):

``` python
CPS_URL_BY_YEAR = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip", 
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
}
```

### CPS Data Structure and Variables

The raw CPS data consists of three interconnected tables: 1. **Person table** (`pppubXX.csv`) - Individual-level demographic and income data (138 variables) 2. **Family table** (`ffpubXX.csv`) - Family-level aggregated data 3. **Household table** (`hhpubXX.csv`) - Household-level characteristics

**Core CPS Variables Extracted:**

**Person-Level Variables** (138 variables total): - **Identifiers**: `PH_SEQ`, `PF_SEQ`, `P_SEQ`, `TAX_ID`, `SPM_ID`, `A_FNLWGT`, `A_LINENO`, `A_SPOUSE` - **Demographics**: `A_AGE`, `A_SEX`, `A_MARITL`, `PRDTRACE`, `PRDTHSP` - **Income Variables**: `WSAL_VAL` (wages), `INT_VAL` (interest), `SEMP_VAL` (self-employment), `FRSE_VAL` (farm), `DIV_VAL` (dividends), `RNT_VAL` (rental), `SS_VAL` (Social Security), etc. - **Disability/Health**: `PEDISEYE`, `PEDISDRS`, `PEDISEAR`, `PEDISOUT`, `PEDISPHY`, `PEDISREM`, `DIS_VAL1`, `DIS_VAL2` - **Other Income**: `OI_VAL`, `CSP_VAL`, `PAW_VAL`, `SSI_VAL`, `VET_VAL`, `WC_VAL` - **Expenses**: `CHSP_VAL` (child support), `PHIP_VAL` (health premiums), `MOOP` (medical out-of-pocket)

**SPM Unit Variables** (72 variables): The Supplemental Poverty Measure (SPM) units provide family-level benefit and tax information including `ACTC`, `BBSUBVAL`, `EITC`, `FEDTAX`, `FICA`, `SNAPSUB`, `WICVAL`, etc.

**Tax Unit Variables** (10 variables): Pre-calculated tax variables including `ACTC_CRD`, `AGI`, `CTC_CRD`, `EIT_CRED`, `FEDTAX_AC`, etc.

### CPS Record Creation and Transformation Process

#### Multi-Level Entity Structure

The CPS processing creates a hierarchical data structure with five entity levels:

1.  **Person** - Individual survey respondents
2.  **Tax Unit** - Filing units for tax purposes\
3.  **Family** - Related individuals living together
4.  **SPM Unit** - Supplemental Poverty Measure units
5.  **Household** - All individuals in a housing unit

#### Tax Unit Creation Logic

Tax units are pre-defined in the CPS data via the `TAX_ID` variable. The process:

1.  **Person-to-Tax Unit Mapping**: Each person has a `TAX_ID` that groups them into tax filing units
2.  **Tax Unit Aggregation**: Tax unit variables are created by summing person-level data:

``` python
def _create_tax_unit_table(person: pd.DataFrame) -> pd.DataFrame:
    tax_unit_df = person[TAX_UNIT_COLUMNS].groupby(person.TAX_ID).sum()
    tax_unit_df["TAX_ID"] = tax_unit_df.index
    return tax_unit_df
```

3.  **Weight Assignment**: Tax unit weights are derived from family weights, ensuring proper statistical representation

#### ID Variable Creation

Complex ID relationships are established:

``` python
# Primary and foreign keys
cps["person_id"] = person.PH_SEQ * 100 + person.P_SEQ
cps["family_id"] = family.FH_SEQ * 10 + family.FFPOS  
cps["household_id"] = household.H_SEQ
cps["person_tax_unit_id"] = person.TAX_ID
cps["tax_unit_id"] = tax_unit.TAX_ID
```

### CPS Variable Processing and Transformations

#### Income Variable Processing

The system applies sophisticated transformations using parameters from `tmd/storage/input/imputation_parameters.yaml`:

**Interest Income Allocation**:

``` python
cps["taxable_interest_income"] = person.INT_VAL * 0.680
cps["tax_exempt_interest_income"] = person.INT_VAL * 0.320
```

**Dividend Income Allocation**:

``` python
cps["qualified_dividend_income"] = person.DIV_VAL * 0.448
cps["non_qualified_dividend_income"] = person.DIV_VAL * 0.552  
```

**Capital Gains Allocation**:

``` python
cps["long_term_capital_gains"] = person.CAP_VAL * 0.880
cps["short_term_capital_gains"] = person.CAP_VAL * 0.120
```

#### Social Security Benefits Processing

Age-based allocation of Social Security benefits:

``` python
MINIMUM_RETIREMENT_AGE = 62
cps["social_security_retirement"] = np.where(
    person.A_AGE >= MINIMUM_RETIREMENT_AGE, person.SS_VAL, 0
)
cps["social_security_disability"] = person.SS_VAL - cps["social_security_retirement"]
```

#### Demographic Variable Processing

**Age Processing**: - CPS reports ages 80-84 as "80" and 85+ as "85" - The system randomly assigns ages 80-84 to avoid bunching:

``` python
cps["age"] = np.where(
    person.A_AGE == 80,
    AGED_RNG.integers(low=80, high=85, endpoint=False, size=len(person)),
    person.A_AGE,
)
```

**Disability Status**:

``` python
DISABILITY_FLAGS = ["PEDIS" + i for i in ["DRS", "EAR", "EYE", "OUT", "PHY", "REM"]]
cps["is_disabled"] = (person[DISABILITY_FLAGS] == 1).any(axis=1)
```

### CPS Integration Logic and Tax Filing Determination

#### Filing Status Determination Process

The integration process occurs in `tmd/datasets/tmd.py`:

1.  **PolicyEngine-US Microsimulation**:

``` python
sim = Microsimulation(dataset=CPS_2021)
nonfiler = ~(sim.calculate("tax_unit_is_filer", period=2022).values > 0)
```

2.  **Filing Rules Applied**: Uses 2022 tax rules (not 2021) to avoid COVID-related anomalies in 2021 data

3.  **Filing Threshold Logic**: Implemented in PolicyEngine-US, based on:

    -   Gross income thresholds
    -   Filing status (Single, Married Filing Jointly, etc.)
    -   Age and blindness status
    -   Dependency status

#### CPS-to-Tax-Calculator Variable Mapping

The transformation to Tax-Calculator format happens in `tmd/datasets/taxcalc_dataset.py`:

**Core Variable Mappings**:

``` python
vnames = {
    "RECID": "household_id",
    "S006": "tax_unit_weight", 
    "E00200": "employment_income",
    "E02100": "farm_income",
    "E00300": "taxable_interest_income",
    "E00650": "qualified_dividend_income",
    "E00900": "self_employment_income",
    "E02400": "social_security",
    "MARS": "filing_status",  # Converted to numeric codes 1-5
    "XTOT": "exemptions_count",
}
```

**Filing Status Conversion**:

``` python
var["MARS"] = (
    pd.Series(pe("filing_status"))
    .map({
        "SINGLE": 1,
        "JOINT": 2, 
        "SEPARATE": 3,
        "HEAD_OF_HOUSEHOLD": 4,
        "SURVIVING_SPOUSE": 5,
    })
    .values
)
```

## PUF Data Processing and CPS Variable Imputation

### PUF Data Sources and Structure

**Data Sources:** - **Primary PUF File**: `tmd/storage/input/puf_2015.csv` (207,697 records) - **Years**: Uses 2015 PUF data as base year, uprated to target years (e.g., 2021) - **Source**: IRS Statistics of Income Public Use File for 2015

**PUF Data Structure:** The PUF contains extensive tax return information with 189+ variables including: - **Tax Variables**: Employment income (E00200), capital gains (E01000), dividends (E00600/E00650), business income (E00900), etc. - **Deduction Variables**: Medical expenses (E17500), state/local taxes (E18400), mortgage interest (E19200), charitable contributions (E19800) - **Credit Variables**: Various tax credits including child tax credit, EITC, education credits - **Identifying Variables**: RECID (record ID), MARS (filing status), XTOT (exemption count) - **Weights**: S006 (sampling weights, scaled by /100 in preprocessing)

### PUF Variable Processing

**Loading and Preprocessing** (`preprocess_puf()` function):

``` python
# Weight rescaling
puf.S006 = puf.S006 / 100

# Filing status mapping
filing_status = puf.MARS.map({
    1: "SINGLE", 2: "JOINT", 3: "SEPARATE", 4: "HEAD_OF_HOUSEHOLD"
})

# Variable transformations (189+ variables mapped to PolicyEngine format)
# Example: QBI calculation
qbi = np.maximum(0, puf.E00900 + puf.E26270 + puf.E02100 + puf.E27200)
```

**Uprating Process** (`uprate_puf.py`): - Uses SOI aggregate targets to uprate 2015 PUF data to target years - **Growth Factors**: Applied using SOI statistical data from `tmd/storage/input/soi.csv` - **Variable-Specific Growth**: Different growth rates for different income types - **Special Handling**: Itemized deductions grow at fixed 2% annually (`ITMDED_GROW_RATE = 0.02`)

### CPS Variable Imputation to PUF Records

**Demographics Data Source**: `tmd/storage/input/demographics_2015.csv` (119,676 records)

**Imputed Variables** (6 demographic variables): 1. **AGEDP1, AGEDP2, AGEDP3**: Ages of up to 3 dependents (categorical age ranges) 2. **AGERANGE**: Primary filer age range (categorical: 1-7) 3. **EARNSPLIT**: How employment income splits between spouses (categorical: 0-4) 4. **GENDER**: Primary filer gender (1=male, 2=female)

**Predictor Variables** (5 tax/filing variables): 1. **E00200**: Employment income 2. **MARS**: Filing status 3. **DSI**: Dependent status indicator 4. **EIC**: Earned Income Credit child count 5. **XTOT**: Total exemptions

### Machine Learning Imputation Details

**Random Forest Configuration**:

``` python
RandomForestRegressor(
    n_estimators=100,           # Default, can be configured
    bootstrap=True,
    max_samples=0.01,           # Uses only 1% of data per tree
    random_state=1928374       # Fixed seed for reproducibility
)
```

**Model Architecture**: - **Sequential Training**: One model per demographic variable, trained in order - **Cascading Predictions**: Each subsequent model uses previously predicted variables as inputs - **Training Process**: 1. Model 1: Predicts AGEDP1 from tax variables 2. Model 2: Predicts AGEDP2 from tax variables + predicted AGEDP1 3. Model 3: Predicts AGEDP3 from tax variables + predicted AGEDP1, AGEDP2 4. And so on...

**Beta Distribution Sampling**:

``` python
# Uses Beta distribution for prediction uncertainty
a = mean_quantile / (1 - mean_quantile)
input_quantiles = random_generator.beta(a, 1, size=tree_predictions.shape[0])
```

**Validation and Quality Control**: - **Fixed Random Seeds**: Two separate seeds for Random Forest (1928374) and Beta sampling (37465) - **Cross-Validation**: Through `solve_for_mean_quantile()` method with binary search - **Integer Encoding**: Automatically detects and handles categorical variables - **Missing Value Handling**: Raises errors for unknown categories during prediction

### PUF Record Enhancement Process

**Step-by-Step Enhancement**:

1.  **Data Loading**: Load PUF 2015 and demographics data
2.  **Subset Identification**:
    -   Records WITH demographics: `puf[puf.RECID.isin(demographics.RECID)]`
    -   Records WITHOUT demographics: Remaining PUF records (\~88,000 records)
3.  **Model Training**: Train on records with both tax and demographic data
4.  **Imputation**: Predict demographics for records lacking them
5.  **Combination**: Merge original + imputed demographic data

**Hierarchical Structure Creation**: The system creates a hierarchical dataset with: - **Tax Units**: Primary filing units - **Persons**: Individual people within tax units (filer, spouse, dependents) - **Age Assignment**: - Filers: Random ages within categorical ranges using `decode_age_filer()` - Dependents: Random ages using `decode_age_dependent()` - **Gender Assignment**: - Primary filer: From imputed GENDER variable - Spouse: 96% opposite gender assumption - Dependents: Random 50/50 assignment

**Earnings Split Implementation**:

``` python
# Splits employment income between spouses based on EARNSPLIT
SPLIT_DECODES = {1: 0.0, 2: 0.25, 3: 0.75, 4: 1.0}
# Filer gets fraction, spouse gets (1 - fraction)
```

## Data Integration and Merging Architecture

### Overall Integration Logic

The TMD dataset creation follows a sophisticated integration architecture:

-   **PUF (Public Use File) 2015**: Detailed tax return information representing tax filers
-   **CPS (Current Population Survey) 2021**: Household demographic and income information representing the broader population including non-filers

### Core Integration Process

The main integration process in `create_tmd_2021()` follows this sequence:

1.  **Data Preparation**: Both PUF and CPS datasets are processed into Tax-Calculator compatible formats
2.  **Filing Status Determination**: CPS records are evaluated to identify non-filers using 2022 filing rules
3.  **Dataset Combination**: PUF filers are combined with CPS non-filers
4.  **Quality Control**: Records with inconsistent characteristics are filtered out
5.  **Weight Scaling**: Population weights are adjusted to achieve proper representation
6.  **Reweighting**: Final optimization to match SOI targets

### Data Source Harmonization

#### Variable Mapping and Transformation

**PUF Variable Processing** (`tmd/datasets/puf.py`):

``` python
# Key financial variables mapped from PUF to common schema
newvars = {
    "employment_income": puf.E00200,
    "qualified_dividend_income": puf.E00650,
    "self_employment_income": puf.E00900,
    "social_security": puf.E02400,
    # ... 70+ additional variables
}
```

**CPS Variable Processing** (`tmd/datasets/cps.py`):

``` python
# Income variables with imputation parameters
cps["employment_income"] = person.WSAL_VAL
cps["taxable_interest_income"] = person.INT_VAL * p["taxable_interest_fraction"]
cps["qualified_dividend_income"] = person.DIV_VAL * p["qualified_dividend_fraction"]
```

### Temporal Integration

#### PUF Uprating Process (`tmd/datasets/uprate_puf.py`)

The temporal alignment involves uprating 2015 PUF data to 2021 using SOI statistics:

**Growth Factor Application**: - **Variable-Specific Growth**: Each income/deduction variable grown using its specific SOI trend - **Population Adjustment**: Growth rates adjusted for changes in filing population - **Special Treatment**: Itemized deductions use constant 2% annual growth rate

``` python
def get_growth(variable, from_year, to_year):
    start_value = get_soi_aggregate(variable, from_year, False)
    end_value = get_soi_aggregate(variable, to_year, False)
    # ... population adjustment logic
    return aggregate_growth / population_growth
```

### Weight Integration and Scaling

#### Initial Weight Processing

**PUF Weights**: - Original PUF weights (S006) divided by 100 to convert to population representation - Adjusted for population growth from 2015 to 2021

**CPS Weights**: - CPS weights scaled by `CPS_WEIGHTS_SCALE = 0.5806` to achieve correct non-filer population - This scaling factor ensures proper representation of the non-filing population

#### Record Reconciliation

**Quality Control Measures**: - **CPS Income Tax Filter**: Removes CPS records with positive income tax (inconsistent with non-filer status) - **Filing Status Validation**: Ensures demographic consistency with filing status - **Income Source Validation**: Verifies retirement contribution limits don't exceed income

**Final Dataset Integration**: 1. **Dataset Combination**: PUF filers + CPS non-filers = Complete population 2. **Reweighting**: Final weights calibrated to SOI aggregate targets using optimization

## Reweighting and Optimization System

### Reweighting Architecture

The reweighting system is built around a PyTorch-based optimization framework that adjusts sampling weights to match Statistics of Income (SOI) targets. Key components include:

-   **Core Module**: `tmd/utils/reweight.py` - Main reweighting implementation
-   **Target Processing**: `tmd/utils/soi_targets.py` - SOI target processing and cleaning
-   **Data Transformation**: `tmd/utils/soi_replication.py` - Data transformation for SOI matching
-   **Configuration**: `tmd/imputation_assumptions.py` - Configuration parameters

### PyTorch Integration

The system leverages PyTorch for: - **Tensor Operations**: Converting pandas DataFrames to torch tensors for efficient computation - **Automatic Differentiation**: Using `requires_grad=True` on weight multipliers for gradient computation - **GPU Support**: Random seed setting for both CPU and CUDA devices - **Optimization**: Adam optimizer with learning rate of 0.1

### Mathematical Framework

The reweighting employs a **multiplicative weight adjustment** approach:

``` python
new_weights = original_weights * clamp(weight_multiplier, min=0.1, max=10.0)
```

Key mathematical components: - **Weight bounds**: 0.1 ≤ multiplier ≤ 10.0 (configurable) - **Relative error minimization**: Using `((outputs + 1) / (targets + 1) - 1)²` formulation - **Weight deviation penalty**: Optional regularization term proportional to weight changes

### SOI Target Structure

The SOI targets are organized with the following structure: - **Year**: Tax year (2015, 2021, etc.) - **Variable**: Income/tax variables (33+ different types) - **Filing Status**: Single, Joint, Head of Household, Separate, All - **AGI lower/upper bound**: Income range boundaries - **Count**: Boolean indicating count vs. amount targets - **Taxable only**: Boolean for taxable vs. all returns - **Value**: Target value to match

#### AGI Ranges

The system uses 19 income ranges from `-∞` to `+∞`:

``` python
INCOME_RANGES = [
    -np.inf, 1, 5e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4, 4e4, 5e4,
    7.5e4, 1e5, 2e5, 5e5, 1e6, 1.5e6, 2e6, 5e6, 1e7, np.inf
]
```

### Optimization Algorithm Details

#### PyTorch Optimization Setup

``` python
# Reproducible random seed
rng_seed = 65748392
torch.manual_seed(rng_seed)
torch.cuda.manual_seed_all(rng_seed)

# Adam optimizer with 0.1 learning rate
optimizer = torch.optim.Adam([weight_multiplier], lr=1e-1)

# 2,000 optimization iterations
for i in tqdm(range(2_000), desc="Optimising weights"):
    # ... optimization loop
```

#### Loss Function Components

**Primary Loss**: Sum of squared relative errors

``` python
loss_primary = ((outputs + 1) / (target_array + 1) - 1) ** 2).sum()
```

**Weight Deviation Penalty** (configurable):

``` python
weight_deviation = (
    (new_weights - original_weights).abs().sum() 
    / original_weights.sum() 
    * weight_deviation_penalty 
    * original_loss_value
)
```

### Performance Monitoring

#### TensorBoard Integration

**Log Directory Structure**:

```         
tmd/storage/output/reweighting/{year}_{timestamp}/
```

**Tracked Metrics** (logged every 100 iterations): - **Summary/Loss**: Total loss value - **Summary/Max relative error**: Maximum absolute relative error across all targets - **Summary/Mean relative error**: Mean absolute relative error - **Estimate/{metric_name}**: Current estimate for each target - **Target/{metric_name}**: Target value for each target\
- **Absolute relative error/{metric_name}**: Per-target absolute relative error

## Validation and Testing Framework

### Testing Architecture

#### Overall Framework Structure

-   **Framework**: pytest-based testing system with comprehensive configuration
-   **Configuration**: `pytest.ini` defines test paths and custom markers
-   **Test Markers**: Multiple specialized markers for different validation categories:
    -   `vartotals`: Variable totals validation
    -   `taxexp`: Tax expenditure validation\
    -   `qbid`: Qualified Business Income Deduction validation
    -   `taxexpdiffs`: Tax expenditure difference validation
    -   `itax`: Income tax validation

#### Test Organization

**Core Test Files**: - `tests/test_tax_revenue.py`: Tax revenue validation against federal agency estimates - `tests/test_tax_expenditures.py`: Tax expenditure validation - `tests/test_variable_totals.py`: Variable totals validation against taxdata benchmarks - `tests/test_area_weights.py`: Area-specific weight validation - `tests/test_tmd_stats.py`: Statistical validation of TMD dataset

### Data Validation Methods

#### Statistical Validation Approaches

-   **Variable Range Validation**: 45% tolerance threshold for variable totals with additional \$30B absolute threshold
-   **Chi-square Testing**: `tmd/areas/chisquare_test.py` implements distribution similarity testing
-   **Bootstrap Sampling**: `tmd/examination/2022/sampling_variability.py` for uncertainty quantification

#### Data Quality Validation

-   **Numerical Integrity**: `np.seterr(all="raise")` converts floating-point exceptions to errors
-   **Weight Validation**: Ensures all sampling weights are positive
-   **Consistency Checks**: Multiple assertions validate data consistency across processing steps

### SOI Replication Testing

#### SOI Comparison Framework

**Key Functions**: - `pe_to_soi()`: Converts PolicyEngine-US data to SOI format - `puf_to_soi()`: Converts PUF data to SOI format\
- `tc_to_soi()`: Converts Tax-Calculator output to SOI format - `compare_soi_replication_to_soi()`: Comprehensive comparison framework

#### Validation Metrics

-   **Absolute Error**: Direct difference between model and SOI values
-   **Relative Error**: Percentage difference calculations
-   **Error Analysis**: Systematic tracking of error patterns across income categories and filing statuses

### Tax Revenue and Expenditure Validation

#### Federal Agency Validation

**Agency Sources**: CBO, JCT, Treasury (TSY) - **Data Files**: `cy23_cbo.csv`, `cy23_jct.csv`, `cy23_tsy.csv`, etc. - **Fiscal-to-Calendar Year Conversion**: `tmd/examination/fy2cy.awk` script

#### Tolerance Levels and Error Thresholds

**Income Tax Revenue**: - Default relative tolerance: 8% (`DEFAULT_RELTOL_ITAX = 0.08`) - Year-specific tolerances: 2023 (12%), 2026 (11%)

**Payroll Tax Revenue**: - Default relative tolerance: 14% (`DEFAULT_RELTOL_PTAX = 0.14`)

#### Tax Expenditure Categories

The system validates seven major tax expenditures: - **CTC**: Child Tax Credit - **EITC**: Earned Income Tax Credit\
- **SSBEN**: Social Security Benefit Exclusion - **NIIT**: Net Investment Income Tax - **CGQD**: Capital Gains/Qualified Dividends preference - **QBID**: Qualified Business Income Deduction - **SALT**: State and Local Tax deduction

### Cross-Model Validation

#### Tax-Calculator Integration

-   **Primary Model**: Tax-Calculator 4.5.0 used for validation
-   **Multiple Datasets**: Validates across different input datasets
-   **Consistency Checks**: Ensures consistent results across different data sources

#### PolicyEngine-US Validation

-   **Data Transformation**: Converts PolicyEngine-US hierarchical data to Tax-Calculator format
-   **Cross-Model Comparison**: Validates estimates between microsimulation models
-   **Reconciliation**: Systematic approach to understanding model differences

### Automated Testing Pipeline

#### Continuous Integration

**GitHub Actions Workflows**: - `.github/workflows/push.yml`: Master branch validation - `.github/workflows/pr.yml`: Pull request validation

#### Build System

**Makefile Targets**: - `make test`: Runs comprehensive test suite with parallel execution - `make tmd_files`: Ensures all data files are current - `make data`: Full validation pipeline (install + build + test)

## Technical Architecture Summary

The Tax-Microdata-Benchmarking project represents a sophisticated data science pipeline that successfully combines:

1.  **Advanced Data Integration**: Merging administrative tax data (PUF) with survey data (CPS) while maintaining statistical integrity
2.  **Machine Learning Enhancement**: Using Random Forest models to impute missing demographic variables with proper uncertainty quantification
3.  **Optimization-Based Reweighting**: Employing PyTorch gradient descent optimization to match official Statistics of Income targets
4.  **Comprehensive Validation**: Multi-layered validation against federal agency estimates, cross-model consistency checks, and statistical quality assurance
5.  **Reproducible Pipeline**: Full automation with proper version control, testing, and documentation

The system maintains statistical integrity through proper weighting, handles missing data through sophisticated imputation, ensures consistency across multiple data entity levels, and produces policy-ready microdata that accurately replicates federal tax revenue patterns while maintaining individual-level detail for distributional analysis.

### Key Strengths

1.  **Data Quality**: Combines strengths of both PUF (tax detail) and CPS (population coverage)
2.  **Statistical Rigor**: Sophisticated imputation and reweighting methods with proper validation
3.  **Policy Relevance**: Accurate replication of federal tax revenue and expenditure estimates
4.  **Technical Innovation**: Modern optimization techniques (PyTorch) applied to traditional survey methodology
5.  **Transparency**: Open-source approach with comprehensive documentation and validation
6.  **Scalability**: Modular design allows for updates and extensions as new data becomes available
7.  **Cross-Model Compatibility**: Supports both Tax-Calculator and PolicyEngine-US microsimulation platforms

This technical architecture enables accurate federal tax policy analysis while maintaining the flexibility to adapt to changing data sources, policy environments, and methodological improvements.