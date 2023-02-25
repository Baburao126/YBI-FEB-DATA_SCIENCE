import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
#1--- load the dataset
df=pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/MultipleDiseasePrediction.csv')
df

#2---preprocess data
le=LabelEncoder()
df["prognosis"]=le.fit_transform(df["prognosis"])
x=df.drop("prognosis",axis=1)
y=df["prognosis"]

df.columns
#3---split the data
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)

#4---train RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,random_state=2529)
clf.fit(x_train,y_train)

#5---prediction
y_pred=clf.predict(x_test)

#6---evaluate model
print("accuracy",accuracy_score(y_test,y_pred))
print("\n classification report\n",classification_report(y_test,y_pred))

