import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv('bbc-text.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.dtypes)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

#Univariate Analysis
plt.figure(figsize=(10,6))
sns.countplot(y='category', data=df, order=df['category'].value_counts().index)
plt.title('Distribution of News Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.grid(True)
plt.savefig('news_category_distribution.png')
plt.show()

#Word Count Analysis
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10,6))
sns.histplot(df['word_count'], bins=30, kde=True)
plt.title('Distribution of Word Counts in Articles')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.savefig('word_count_distribution.png')
plt.show()

#BiVariant Analysis 

#Average Word Count per Category
avg_word_count = df.groupby('category')['word_count'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=avg_word_count.values, y=avg_word_count.index)
plt.title('Average Word Count per News Category')
plt.xlabel('Average Word Count')
plt.ylabel('Category')
plt.grid(True)
plt.savefig('avg_word_count_per_category.png')
plt.show()

#Boxplot of Word Count by Category
filtered_df = df[df['category'].isin(['tech', 'sport'])]
plt.figure(figsize=(10,6))
sns.boxplot(x='category', y='word_count', data=filtered_df)
plt.title('Boxplot of Word Count: Tech vs Sport Articles')
plt.grid(True)
plt.savefig('boxplot_word_count_tech_sport.png')
plt.show()

#Violin Plot of Word Count by Category
plt.figure(figsize=(10,6))
sns.violinplot(x='word_count', y='category', data=df, hue='category')
plt.title('Violin Plot of Word Count vs Category')
plt.grid(True)  
plt.savefig('violin_plot_word_count_category.png')
plt.show()

# This removes any row where the 'text' or 'category' is missing
df = df.dropna(subset=['text', 'category'])  
df = df.reset_index(drop=True)                                                  
print("Data cleaning completed. Any rows with missing 'text' or 'category' have been removed.")

