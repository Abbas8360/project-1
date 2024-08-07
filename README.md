Task 1: Exploratory Data Analysis on a Dataset
Objective: Perform exploratory data analysis (EDA) on a given dataset to understand its structure and basic statistics.

Steps:
1. Load the Dataset: Use Pandas to load a CSV file into a DataFrame.
2. Data Cleaning: Check for missing values, duplicates, and data types. Handle missing values and duplicates appropriately.
3. Summary Statistics: Use Pandas and NumPy to compute summary statistics (mean, median, mode, standard deviation) for numerical columns.
4. Visualization: Use Seaborn to create visualizations such as histograms, box plots, and pair plots to explore the distribution and relationships between variables.

Example Dataset: Titanic dataset.

 Task 2: Time Series Analysis
Objective: Analyze a time series dataset to identify trends, seasonality, and patterns.

Steps:
1. Load the Dataset: Use Pandas to load a time series dataset.
2. Date Handling: Ensure the date column is in datetime format and set it as the index.
3. Resampling and Aggregation: Use Pandas to resample the data (e.g., daily to monthly) and calculate aggregate metrics.
4. Trend Analysis: Use NumPy to calculate rolling means and standard deviations.
5. Visualization: Use Seaborn to create line plots, heatmaps, and seasonal plots.

Example Dataset: Stock prices.

 Task 3: Data Manipulation and Feature Engineering
Objective: Perform data manipulation and create new features from an existing dataset.

Steps:
1. Load the Dataset: Use Pandas to load a dataset.
2. Data Transformation: Use Pandas and NumPy to transform columns (e.g., log transformation, normalization).
3. Feature Engineering: Create new features based on existing columns (e.g., extracting year/month from a date, creating interaction terms).
4. Grouping and Aggregation: Use Pandas to group data by a categorical variable and compute aggregated statistics.
5. Visualization: Use Seaborn to visualize the new features and their relationships with other variables.

Example Dataset: Housing prices, 

 Task 4: Statistical Analysis and Hypothesis Testing
Objective: Perform statistical analysis and hypothesis testing on a dataset to draw conclusions.

Steps:
1. Load the Dataset: Use Pandas to load a dataset.
2. Descriptive Statistics: Use Pandas and NumPy to compute descriptive statistics.
3. Hypothesis Testing: Perform t-tests, chi-square tests, and ANOVA using relevant libraries.
4. Correlation Analysis: Use Seaborn to create heatmaps and pair plots to visualize correlations between variables.
5. Visualization: Use Seaborn to create visualizations that support the statistical analysis (e.g., bar plots, violin plots).

Example Dataset: Customer satisfaction survey data, 

 Task 5: Machine Learning Preparation and Visualization
Objective: Prepare a dataset for machine learning and visualize key aspects of the data.

Steps:
1. Load the Dataset: Use Pandas to load a dataset.
2. Data Preprocessing: Handle missing values, encode categorical variables, and scale numerical features using Pandas and NumPy.
3. Feature Selection: Use correlation analysis and feature importance metrics to select relevant features.
4. Split the Data: Split the data into training and testing sets.
5. Visualization: Use Seaborn to visualize the feature distributions, relationships between features, and target variable distributions.

Example Dataset: loan default dataset.




============================================================
task 1
 python code .(using google colab for Run)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/content/titanic.csv')
df.head()
df.info()
df.describe()
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# Print initial shape of the dataset
print(f"Original dataset shape: {df.shape}")

# Remove rows with any missing values
df_cleaned = df.dropna()

# Print the shape of the cleaned dataset
print(f"Cleaned dataset shape: {df_cleaned.shape}")

# Verify that there are no missing values in the cleaned DataFrame
missing_values_cleaned = df_cleaned.isnull().sum()
print("Missing Values in Cleaned DataFrame:\n", missing_values_cleaned)
df.drop_duplicates(inplace=True)
df.head()

# Set the style for the plots
sns.set(style="whitegrid")

# Plot histograms for numerical columns
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
df[numerical_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle('Histograms of Numerical Columns')
plt.show()
==================================================================================================
task 2
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plots
sns.set(style="whitegrid")

# List of numerical columns to plot
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot histograms with KDE for each numerical column
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=20)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.suptitle('Histograms with KDE of Numerical Columns', y=1.02)
plt.show()
# Box plot for Age by Pclass
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Box Plot of Age by Pclass')
plt.show()

# Box plot for Fare by Pclass
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Box Plot of Fare by Pclass')
plt.show()

# Pair plot for numerical columns
sns.pairplot(df[numerical_cols])
plt.suptitle('Pair Plot of Numerical Columns', y=1.02)
plt.show()
# Distribution plot for Age
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.show()

# Distribution plot for Fare
plt.figure(figsize=(12, 6))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Distribution of Fare')
plt.show()
# Count plot for Pclass
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Count of Passengers by Pclass')
plt.show()

# Count plot for Survived
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df)
plt.title('Count of Passengers by Survival')
plt.show()
# Compute correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()
======================================================================================
task 3
import pandas as pd
# Load the dataset
df = pd.read_csv('/content/housing.csv')
df.head()
deletemissing = df.dropna()

import numpy as np

# Apply log transformation to the Price column
df['Price_log'] = np.log1p(df['Price'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Size_normalized'] = scaler.fit_transform(df[['Size']])
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df.head()
df['Price_per_Size'] = df['Price'] / df['Size']
df.head()
# Group by Location and compute average Price and Size
location_stats = df.groupby('Location').agg({
    'Price': ['mean', 'median', 'std'],
    'Size': ['mean', 'median', 'std'],
    'Price_per_Size': ['mean', 'median']
})
import seaborn as sns
import matplotlib.pyplot as plt

# Plot distribution of the log-transformed Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Price_log'], kde=True)
plt.title('Distribution of Log-Transformed Price')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.show()
# Scatter plot of Price vs. Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size', y='Price', data=df)
plt.title('Price vs. Size')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()
# Box plot of Price by Location
plt.figure(figsize=(12, 8))
sns.boxplot(x='Location', y='Price', data=df)
plt.xticks(rotation=45)
plt.title('Price by Location')
plt.xlabel('Location')
plt.ylabel('Price')
plt.show()
# Heatmap of aggregated statistics by Location
plt.figure(figsize=(12, 8))
sns.heatmap(location_stats, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Aggregated Statistics by Location')
plt.show()
df = pd.read_csv('/content/customer_satisfaction.csv')
df.head()

count_missing = df.isnull().sum()
print(count_missing)
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Check for duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Check data types
print("Data Types:\n", df.dtypes)
# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (if any)
df.fillna({
    'Age': df['Age'].median(),  # Impute missing Age with median
    'Satisfaction': df['Satisfaction'].mode()[0]  # Impute missing Satisfaction with mode
}, inplace=True)
import numpy as np

# Summary statistics for numerical columns
print("Summary Statistics:\n", df.describe(include=[np.number]))

# Mode for categorical columns
print("Mode:\n", df.mode().iloc[0])
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plots
sns.set(style="whitegrid")

# Histogram for Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box Plot for Satisfaction by Product
plt.figure(figsize=(12, 6))
sns.boxplot(x='Product', y='Satisfaction', data=df)
plt.title('Box Plot of Satisfaction by Product')
plt.xlabel('Product')
plt.ylabel('Satisfaction')
plt.show()

# Count Plot for Product
plt.figure(figsize=(8, 6))
sns.countplot(x='Product', data=df)
plt.title('Count of Each Product')
plt.xlabel('Product')
plt.ylabel('Count')
plt.show()

# Pair Plot for numerical columns
sns.pairplot(df[['Age', 'Satisfaction']])
plt.suptitle('Pair Plot of Age and Satisfaction', y=1.02)
plt.show()
# Create a feature for Age Group
bins = [0, 18, 30, 45, 60, 100]
labels = ['18-24', '25-29', '30-44', '45-59', '60+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Create an interaction feature between Age and Satisfaction
df['Age_Satisfaction_Interaction'] = df['Age'] * df['Satisfaction']
# Group by Product and compute average satisfaction
product_stats = df.groupby('Product')['Satisfaction'].agg(['mean', 'median', 'std'])
print("Product Statistics:\n", product_stats)
# Histogram for Age Groups
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', data=df)
plt.title('Count of Customers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Scatter Plot for Age vs Satisfaction with Age Group as Hue
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='Satisfaction', hue='AgeGroup', data=df)
plt.title('Age vs Satisfaction by Age Group')
plt.xlabel('Age')
plt.ylabel('Satisfaction')
plt.show()
df=pd.read_csv('/content/loan_default.csv')
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Handle missing values
df.fillna({
    'Amount': df['Amount'].median(),         # Impute missing Amount with median
    'Term': df['Term'].mode()[0],            # Impute missing Term with mode
    'InterestRate': df['InterestRate'].median()  # Impute missing InterestRate with median
}, inplace=True)
# Encode categorical variable 'Term' (if it were categorical)
df['Term'] = df['Term'].astype('category').cat.codes
from sklearn.preprocessing import StandardScaler

# Scale numerical features
scaler = StandardScaler()
df[['Amount', 'InterestRate']] = scaler.fit_transform(df[['Amount', 'InterestRate']])
# Compute correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Visualize correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
from sklearn.ensemble import RandomForestClassifier

# Define features and target variable
X = df[['Amount', 'Term', 'InterestRate']]
y = df['Default']

# Fit RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importances)
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
# Histogram for Amount
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], kde=True)
plt.title('Distribution of Loan Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Histogram for Interest Rate
plt.figure(figsize=(10, 6))
sns.histplot(df['InterestRate'], kde=True)
plt.title('Distribution of Interest Rate')
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')
plt.show()
# Scatter plot for Amount vs Interest Rate
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Amount', y='InterestRate', hue='Default', data=df, palette='viridis')
plt.title('Amount vs Interest Rate')
plt.xlabel('Amount')
plt.ylabel('Interest Rate')
plt.legend(title='Default')
plt.show()
# Count plot for Default
plt.figure(figsize=(8, 6))
sns.countplot(x='Default', data=df, palette='pastel')
plt.title('Distribution of Loan Default')
plt.xlabel('Default')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
# Count the occurrences of each category in the 'Default' column
default_counts = df['Default'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(default_counts, labels=['Non-Default', 'Default'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
plt.title('Proportion of Loan Defaults')
plt.show()
