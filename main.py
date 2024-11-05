#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read the csv file
df = pd.read_csv('mushrooms.csv')

#Take a look at the data
print(df.head())

#Find missing values (none)
print(df.isnull().sum())

#Find distribution of poisonous and edible mushrooms
print(df['class'].value_counts())

#Use Label Encoding to transform data into numbers
for columns in df.columns:
  label_encoder = LabelEncoder()
  df[columns] = label_encoder.fit_transform(df[columns])

#Look at encoded data
print(df.head())

#Split data into testing and training sets

df = df.sample(frac=0.1) #Shuffle the dataset

#Choose ratio of training to test data (80% to 20%)
training_rows = int(df.shape[0] * 0.8)

#X sets include the independent variables, Y sets include the outcome (poisonous or edible)
X_train = df.iloc[:training_rows, 1:]
X_test = df.iloc[training_rows:, 1:]

Y_train = df.iloc[:training_rows, 0]
Y_test = df.iloc[training_rows:, 0]

#Create Model
model = DecisionTreeClassifier().fit(X_train, Y_train)

#Evaluate Model
print("Accuracy: ", model.score(X_test, Y_test))

#Visualization
fig = plt.figure(figsize=(15,15))
plot_tree(model, feature_names=X_train.columns, rounded = True, filled = True) 
fig.savefig('tree.png')