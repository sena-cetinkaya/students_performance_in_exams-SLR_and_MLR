# Import the libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the dataset.
data = pd.read_csv("StudentsPerformance.csv")

# Display the first few rows of the data.
print(data.head)

# Learning columns.
print(data.columns)

# Data types of columns.
print(data.dtypes)

# Learning the size of the dataset.
print(data.shape)

# Check for missing values.
print(data.isnull().sum())

# Summary statistics.
print(data.describe)

print(data.info)

# Analyzes of categorical data
par_lv_of_edu = data["parental level of education"].value_counts()
print(par_lv_of_edu)

female_male_ratio = data["gender"].value_counts()
print(female_male_ratio)

race_ethnicity = data["race/ethnicity"].value_counts()
print(race_ethnicity)

lunch = data["lunch"].value_counts()
print(lunch)

test_pre_course = data["test preparation course"].value_counts()
print(test_pre_course)

# DATA VISUALIZATION
# Ratios of male and female data relative to each other.
female_male_ratio.plot.pie(figsize=(4, 4), title="Men - Women", ylabel="", colors=['pink', 'lightblue'], autopct="%.0f%%")
plt.show()

# Bar chart of Parents' Education Level.
par_lv_of_edu.plot.bar(figsize=(8,4), title="Parental Level of Education",ylabel="", color="#E3CEF6")
plt.show()

# Bar chart by race/ethnicity.
race_ethnicity.plot.bar(figsize=(6,4), title="Race Ethnicity",ylabel="", color="#A9D0F5")
plt.show()

# Proportions of lunch categories.
lunch.plot.pie(figsize=(4,4), title="Lunch", ylabel="", autopct="%.0f%%", colors=["#F7BE81","#CEF6F5"])
plt.show()

# Proportions of test preparation course categories.
test_pre_course.plot.pie(figsize=(4,4), title="Test Preparation Course", ylabel="", autopct="%.0f%%", colors=["#F78181","#ACFA58"])
plt.show()

# Proportions of Categorical Data
plt.subplot(2, 3, 1)
female_male_ratio.plot.pie(figsize=(4, 4), title="Men - Women", ylabel="", colors=['pink', 'lightblue'], autopct="%.0f%%")

plt.subplot(2, 3, 2)
lunch.plot.pie(figsize=(4,4), title="Lunch", ylabel="", autopct="%.0f%%", colors=["#F7BE81","#CEF6F5"])

plt.subplot(2, 3, 3)
test_pre_course.plot.pie(figsize=(4,4), title="Test Preparation Course", ylabel="", autopct="%.0f%%", colors=["#F78181","#ACFA58"])

plt.subplot(2, 3, 4)
par_lv_of_edu.plot.bar(figsize=(8,4), title="Parental Level of Education",ylabel="",color="#E3CEF6")

plt.subplot(2, 3, 6)
race_ethnicity.plot.bar(figsize=(6,4), title="Race Ethnicity",ylabel="", color="#A9D0F5")

plt.suptitle("Proportions of Categorical Data")
plt.show()

# Relationship between numerical data.
# Scatter plot between math score and reading score.
sns.scatterplot(x="math score", y="reading score",data=data, color="lightblue")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Math Score - Reading Score")
plt.show()

# Scatter plot between math score and writing score.
sns.scatterplot(x="math score", y="writing score",data=data, color="pink")
plt.xlabel("Math Score")
plt.ylabel("Writing Score")
plt.title("Math Score - Writing Score")
plt.show()

# Scatter plot between writing score and reading score.
sns.scatterplot(x="reading score", y="writing score",data=data, color="#F5DA81")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.title("Reading Score - Writing Score")
plt.show()

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
sns.scatterplot(x="math score", y="reading score",data=data, color="lightblue")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Math Score - Reading Score")

plt.subplot(1, 3, 2)
sns.scatterplot(x="math score", y="writing score",data=data, color="pink")
plt.xlabel("Math Score")
plt.ylabel("Writing Score")
plt.title("Math Score - Writing Score")

plt.subplot(1, 3, 3)
sns.scatterplot(x="reading score", y="writing score",data=data, color="#F5DA81")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.title("Reading Score - Writing Score")

plt.suptitle("Proportions of Numerical Data")
plt.show()

# Simple Linear Regression
# As can be seen from the graphs above, there is a linear relationship between writing-score, reading-score and math-score.
# Let's examine the success percentage of the model by creating a simple linear regression model
# between reading-score and writing-score from this data.

X = data[["reading score"]]
y = data[["writing score"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2 Score: ", r2_score(y_test, y_pred)*100)

#Relationship between actual data and predicted data.
plt.scatter(X_train, y_train, color="pink")
X_train_pred = model.predict((X_train))
plt.scatter(X_train, X_train_pred, color="lightblue")
plt.title('Reading Score - Writing Score')
plt.xlabel('Reading score')
plt.ylabel('Writing score')
plt.show()

#Multiple Linear Regression
#Now let's create a multiple linear regression model using math-score, reading-score and writing-score data.

X = data.iloc[:,6:8]
y = data['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("R2 Score: ", r2_score(y_test, y_pred)*100)
