import pandas as pd
data= pd.read_csv(r'heart.csv')
data.isnull().sum()
data_dup = data.duplicated().any()
data_dup
data = data.drop_duplicates()
data_dup = data.duplicated().any()
data_dup
cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
print(cate_val)
print(cont_val)
cate_val
cont_val
cate_val
data['sex'].unique()
cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns=cate_val,drop_first=True)
data.head()
data.head()
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])
data.head()
X = data.drop('target',axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
y_test
data.head()
data.head()
from sklearn.linear_model import LogisticRegression
Log = LogisticRegression()
Log.fit(X_train,y_train)
y_pred1 = Log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
