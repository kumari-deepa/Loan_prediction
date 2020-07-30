
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
#Let's start with importing necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score



path='C:/Users/kumar/Desktop/loan prediction/train.csv'
df = pd.read_csv(path)
f1='C:/Users/kumar/Desktop/loan prediction/test.csv'
df1=pd.read_csv(f1)
data = df.append(df1,sort=False)
data.head(20)
data.describe()
data['Dependents']=data['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
data['Dependents'] = data['Dependents'].fillna(value=data['Dependents'].mean())
data['Dependents']=data['Dependents'].astype(int)
data['Gender'].fillna(value = 'Female',inplace = True)
data['Gender']=data['Gender'].map({'Male': 0,'Female' :1})
data['Married'] = data['Married'].map({'Yes': 0,'No' :1})
data['Married']=data['Married'].replace(np.NaN,0)
data['Married'].fillna(value = data['Married'].mean())
data['Self_Employed']=data['Self_Employed'].map({'Yes': 0,'No' :1})

data['Self_Employed'].fillna(value = data['Self_Employed'].mean(),inplace = True)
data['Loan_ID'] = data['Loan_ID'].str[-4:]
data['Loan_ID'] = data.Loan_ID.astype(int)
data['Loan_Amount_Term']=data['Loan_Amount_Term'].replace(np.NaN,0)
data['Loan_Amount_Term']=data['Loan_Amount_Term'].fillna(value = data['Loan_Amount_Term'].mean())
data['LoanAmount']=data['LoanAmount'].fillna(value = data['LoanAmount'].mean())
data['Credit_History']=data['Credit_History'].replace(np.NaN,0)
data['Credit_History']=data['Credit_History'].fillna(value = data['Credit_History'].mean())
data.Married = data.Married.astype(int)
data.Self_Employed = data.Self_Employed.astype(int)
data.Credit_History = data.Credit_History.astype(int)
data['Property_Area'] = data['Property_Area'].map({'Semiurban': 0 , 'Urban' : 1 , 'Rural' : 2})
data['Education'] =data['Education'].map({'Graduate': 0 , 'Not Graduate' : 1})
data.Loan_Amount_Term = data.Loan_Amount_Term.astype(int)

#data = data.drop(columns = ['Loan_ID'])
x=data.drop(columns = ['Loan_Status'])
y=data['Loan_Status']
y = y.map({'Y': 0 , 'N' : 1})
y=y.replace(np.NaN,0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state= 355)
knn = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size =18, n_neighbors =10)

knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(knn, open(filename, 'wb'))

