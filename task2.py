import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

print ("1. Summary Statistics: ")
# Generate summary statistics for numeric columns (mean, median, std, etc.)
print ("\nNumeric Statistics: ")
print ("df.describe()")

# Generate summary statistics for categorical columns
print ("\n Categorical Statistics: ")
print(df.describe(include=['O']))

print ("\nHistograms and Boxplots: ")
# Histogram for Age distribution
plt.figure (figsize=(8, 4))
sns.histplot (df['Age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title ('Distibution of Passenger Ages')
plt.xlabel ('Age')
plt.ylabel ('Count')
plt.show()   # Close the window to proceed to the next plot

# Bloxplot to check for outliers in Fare
plt.figure (figsize=(8, 4))
sns.boxplot(x=df['Fare'], color='lightgreen')
plt.title('Boxplot of Fares (Outlier Detection)')
plt.show()

print ("\n3. Correlation Matrix: ")
# Use correlation matrix for feature relationships
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure (figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title ('Correlation Matrix of Numeric Features')
plt.show()

print("\n4. Patterns, Trends, and Inferences: ")
# Idebtufy patterns: Survival rate by Gender
plt.figure (figsize=(6, 4))
sns.countplot (data=df, x='Survived', hue='Sex', palette='Set2')
plt.title ('Survival Count by Gender (0 = No, 1 = Yes)')
plt.show()

# Identify patterns: Survival rate by Passenger Class
plt.figure (figsize=(6, 4))
sns.countplot (data=df, x='Survived', hue='Pclass', palette='Set1')
plt.title ('Survival Count by Passenger Class')
plt.show()