import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


df = pd.read_csv('loan.csv')


df[['Gender','Married','Self_Employed']]=df[['Gender','Married','Self_Employed']].fillna('UNKNOWN')


# drop unnesscery columns
df = df.drop('Loan_ID',axis=1)

imputer = SimpleImputer()
df[['Loan_Amount_Term','LoanAmount']] = imputer.fit_transform(df[['Loan_Amount_Term','LoanAmount']])

imputer2 = SimpleImputer(strategy='most_frequent')
df[['Credit_History']] = imputer2.fit_transform(df[['Credit_History']])


#encoding

encoder = LabelEncoder()
encoded_list = ['Gender','Married','Self_Employed','Education','Property_Area']
for column in encoded_list:
    df[column] = encoder.fit_transform(df[column])


#scaling
scaler = MinMaxScaler()
scaled_list = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
df[scaled_list] = scaler.fit_transform(df[scaled_list])


#train test split
features = df.drop('Loan_Status' , axis=1)
target = df['Loan_Status']

x_train , x_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42) 


#models training and testing

#decision tree
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)



#SVC
model_2 = SVC()
model_2.fit(x_train,y_train,)
y_pred_2 = model_2.predict(x_test)


#Quadratic Discriminant Analysis
model_3 = QuadraticDiscriminantAnalysis()
model_3.fit(x_train,y_train)
y_pred_3 = model_3.predict(x_test)



print('accuracy for the three models respectively:\n\ndecision tree: ' + str(accuracy_score(y_test,y_pred))
      +'\nSVC: ' + str(accuracy_score(y_test,y_pred_2))
      +'\nQuadratic Discriminant Analysis: ' + str(accuracy_score(y_test,y_pred_3)))

print('\nprecision for the three models respectively:\n\ndecision tree: ' + str(precision_score(y_test,y_pred))
      +'\nSVC: ' + str(precision_score(y_test,y_pred_2))
      +'\nQuadratic Discriminant Analysis: ' + str(precision_score(y_test,y_pred_3)))

