import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io

csv_data = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Fare,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,7.25,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,71.2833,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,7.925,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,53.1,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,8.05,S
6,0,3,"Moran, Mr. James",male,,0,0,8.4583,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,51.8625,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,21.075,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,11.1333,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,30.0708,C
11,1,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,16.7,S
12,1,1,"Bonnell, Miss. Elizabeth",female,58,0,0,26.55,S
13,0,3,"Saundercock, Mr. William Henry",male,20,0,0,8.05,S
14,0,3,"Andersson, Mr. Anders Johan",male,39,1,5,31.275,S
15,0,3,"Vestrom, Miss. Hulda Amanda Adolfina",female,14,0,0,7.8542,S
16,1,2,"Hewlett, Mrs. (Mary D Kingcome) ",female,55,0,0,16,S
17,0,3,"Rice, Master. Eugene",male,2,4,1,29.125,Q
18,1,2,"Williams, Mr. Charles Eugene",male,,1,0,13,S
19,0,3,"Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)",female,31,1,0,18,S
20,1,1,"Spencer, Mrs. William Augustus (Marie Eugenie)",female,,0,0,146.5208,C
21,0,2,"Fynney, Mr. Joseph J",male,35,0,0,26,"""

df = pd.read_csv(io.StringIO(csv_data))

print("----------- 1. Raw Data Info -----------")
print("Original shape:", df.shape)
df.info()
print("\n")

cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1, errors='ignore')

print("----------- 2. Data After Dropping Unnecessary Columns -----------")
print("Shape after dropping columns:", df.shape)
print(df.head())
print("\n")

df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

print("----------- 3. Data After Creating Dummy Variables -----------")
print("Shape after creating dummies:", df.shape)
print(df.head())
print("\n")

df['Age'] = df['Age'].interpolate()

print("----------- 4. Final Cleaned Data Info -----------")
df.info()
print("\n")

X = df.drop('Survived', axis=1)
y = df['Survived']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("----------- 5. Final Dataset Shapes -----------")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nPreprocessing complete. Data is ready for model training.")