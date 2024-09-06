## Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
 Hardware – PCs
 Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
Step 1 : Import the standard libraries such as pandas module to read the corresponding csv file.

Step 2 : Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

Step 3 : Import LabelEncoder and encode the corresponding dataset values.

Step 4 : Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

Step 5 : Predict the values of array using the variable y_pred.

Step 6 : Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

Step 7 : Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8: End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:bala murugan s
RegisterNumber:212223230027
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

data1.isnull()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklea.metrics import accuracy_scorern
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![Screenshot 2024-09-06 101533](https://github.com/user-attachments/assets/f2274fd6-c392-4e41-9565-38bea3069e23)
![Screenshot 2024-09-06 101539](https://github.com/user-attachments/assets/fa4a1681-dea0-4f7a-82f4-96a2fbe27799)
![Screenshot 2024-09-06 101545](https://github.com/user-attachments/assets/8cfd0483-c1d3-41d9-8acf-a8206485dafd)
![Screenshot 2024-09-06 101552](https://github.com/user-attachments/assets/5d85a6c1-e84f-49e3-9309-e06430e66015)
![Screenshot 2024-09-06 101557](https://github.com/user-attachments/assets/6ea7db5a-125a-4966-a98d-3879540a7a12)
![Screenshot 2024-09-06 101604](https://github.com/user-attachments/assets/8dd76b5f-b2d5-4994-b4cd-870a8f4309c7)
![Screenshot 2024-09-06 101610](https://github.com/user-attachments/assets/ae68581b-ee99-401f-8a03-c183dee17125)
![Screenshot 2024-09-06 101614](https://github.com/user-attachments/assets/f3842248-2219-4e71-b75c-a54614879a06)
![Screenshot 2024-09-06 101632](https://github.com/user-attachments/assets/26bca60a-b260-4374-941e-0c07d243e612)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
