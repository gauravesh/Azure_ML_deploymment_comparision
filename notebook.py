Import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

df=pd.read_csv("Users/gouraveshsharma/titanic.csv")
df.head(5)

# Step 3: Preprocess the data
# Drop unnecessary columns
titanic_data = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# Fill missing values with median for 'Age' and mode for 'Embarked'
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# Convert categorical features to numerical
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])


# Step 4: Split the data into training and testing sets
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# Step 6: Make predictions on the testing set
y_pred = clf.predict(X_test)


# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

auc_weighted = roc_auc_score(y_test, y_pred, average='weighted')
print("Weighted AUC:", auc_weighted)
