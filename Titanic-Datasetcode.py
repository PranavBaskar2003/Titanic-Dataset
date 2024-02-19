import xdrlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv("c:/Users/Pranav/Documents/Titanic-Dataset.csv")
columns=data.columns.to_list()
print(columns)
print(data.describe().T)
print(data['Survived'].value_counts())
print(data['Pclass'].value_counts()) 
print(data['Sex'].value_counts())
print(data['SibSp'].value_counts())
print(data['Parch'].value_counts())
data['Fare_Category'] = pd.cut(data['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid','High_Mid','High'])
print(data['Fare_Category'].value_counts())
print(pd.crosstab(data['Fare_Category'],data['Survived']))
sns.histplot(data, x='Age',hue='Survived',palette='magma', kde=True)
plt.show()
sns.catplot(data,x='Pclass',hue='Sex',palette='magma',col='Survived',kind='count')
plt.show()
sns.countplot(data,x='Embarked',hue='Survived',palette='magma')
plt.show()
sns.boxplot(data,x='Pclass',y='Age',palette='magma')
plt.show()
print(data.drop('Fare_Category',axis=1,inplace=True))
print(data.info())
print(data.Embarked.fillna(data.Embarked.mode()[0],inplace=True))
print(data.info())
print(data.Cabin.fillna('NA',inplace=True))
print(data.info())
print(data.Name)
data['Salutation']=data.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())
data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
print(data['Age'].fillna(data['Age'].median(), inplace=True))
data['Age_group']=pd.cut(data['Age'],bins=[0,18,35,50,100],labels=['0-18','19-35','36-50','51+'])
data['Family_Size']=data['SibSp']+data['Parch']+1
print(data.Family_Size.value_counts())
data['Fare_range']=pd.qcut(data['Fare'],q=4,labels=['Low','Medium','High','Veryhigh'])
print(data['Fare_range'])
print(data.head())
print(data['Salutation'].unique().tolist())
columns=data.columns.to_list()
print(columns)
print(data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True))
sns.barplot(data, x='Age_group',y='Survived',palette='magma')
plt.show()
sns.barplot(data,x='Embarked',y='Survived',palette='magma')
plt.show()
sns.barplot(data, x='Family_Size',y='Survived',palette='magma')
plt.show()
sns.barplot(data, x="Fare_range",y="Survived",palette='magma')
plt.show()
