# Fault Detection & Anomaly Detection

***
## Step1:dd Data Processing
1. upload the data
   we have our data in space seperated forum and text file
    thus we need to trasnfer it into csv
2. 

> `sep= " "`  and `delim_whitespace=True `
> - `sep= " "` is space as separator
>   -  only value one space as separator (not nulti spaces or tab)
> - `delim_whitespace=True`
>   - more fexible and valid for most cases

> `names = [...]`
>  - naming the header for no-header dataset manually

> `df.concat()` 
> - combining two DataFrame column-wise 
> - `axis=1` column wise: side by side
> - `axis=0` row wise: stacking on top
> - e.g `pd.concat([df_data, df_label], axis=1)`

***
## Step2: Exploratory data analysis
### 2.1: check for Missing Values
The missing values will cause error, bias, and inaccurate
- also affect the feature importance and correlation
  - feature importance: to see if the feature is useful for the model or not

1. count the missing values in each column
2. calculate the percentage
3. sort and display the columns with missing data

> `data.sort_values`

### 2.2: handle missing value
two options for handling
1. drop the feathre with too many missing values (missing rate > 50%)
2. impute missimg values and fill them with 
   1. mean 
   2. median...etc
> `.index`
> - give the list of feature names
> - can be operate with `.drop()` for only dropping the expect column names
>   - won't drop the entire DataFrame

> `.fillna()`
> - filling the missing value with a specified value
> - could be `.fillna(value)`, `.fillna(df.mean)`, and `.fillna(df,median)`
> `inplace=True`
> - modefied the DataFrame directly instead of creating the new one
>  --> the DataFrame remains unchanged
> - !!!! this modified the Dataframe directly, but returns `None`

> `index=False`
> - since our index without meaningful identifier
> - removing the row label

### 2.3: check feature distruibution (outlier and skewness)
outliers: extreme values that exceed the proper range
- might distort model traning and lead to poor generalization
skewness: see if the data is normally distributed or imbalanced
- if is imbalanced --> log transformation or normalization

#### outliers 
**IQR (Interquartile Range)**
- Q1 (25th percentile): lower quartile
- Q3 (75th percentile): upper quartile
- IQR = Q3 - Q1
- Outliers: 
  - below Q1 - 1.5*IQR
  - above Q3 + 1.5*IQR
IQR is works well for moderately large dataset
but not well with the dataset with too much features such as isolation forest...

> `.iloc[row selection, column selection]`
> - indexing method (position based)
> - integer position (:)
> `().sum`
> - count the num of `True` 

**Capping (Winsorization)**
use to limit extreme values by replacing then with upper and lower threshold values
- could keep all the data points
- reduce the impact of extreme values

> `.clip(lower, upper)`
> - if the value is below `lower_bound`, it is set to `lower_bound`
> - `axis = 1`: the operation is applied to columns

#### Skewness
- sknewness = 0: perfectly symmetric
- skew > 0: Positive skewness, long tail on the right side
  - high outliers exist
  - oftern need log transformation 
- skew < 0: Negative skewness, long tail on the left side
  - low outliers exist
  - often need square transformation 

|Skewness Range |  | Transformation|
| --|--|--|
| skew < -1 | Highly left-skewed | Yeo-Johnson Transformation|
| -1 <= skew <= 1| Near symmetric | N/A|
| 1 < skew <= 1.5| Moderate right-skewed| square root transformation (`sqrt(x)`)|
| skew > 1.5 | Highly right-skewed| log transformation (`log1p(x)`)|

> `.all()` in pandas
> - `df["Features"].all()`
> returns True if all values in "Feature" are non-zero
> returns False if any value is 0 or false


### 2.4: Correlations Analysis
#### correlation
a statistical measure that describe if two variables are related
- 1: perfectly positive correlation
- -1: perfectly negative correlation
- 0: no correlation

in EDA correlation can used to avoid overfitting, feature selection
muticollinearity: when two or more features are highly correlated, it can cause instability in model training

#### heatmap
visualize the correlation matrix using a color-coded grid

#### Identify features most correlated with `Label`
`label` is the target variable
features are the variables that can be used to predict the label

"the correlation with label is the most important" for how well the feature is correlates with the label

#### remove highly correlated features
as called Multicollinearity check




### 2.5: save cleaned Dataset

