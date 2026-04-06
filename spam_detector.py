import pandas as pd
df=pd.read_csv("spam.csv",sep="\t",header=None,names=["lable","message"])
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
X=df["message"]
y=df["lable"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer = CountVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_vec,y_train)

accuracy=model.score(X_test_vec,y_test)
print(f"Maodel Accuracy : {accuracy*100:.2f}%")

def predict(message):
    msg_vec= vectorizer.transform([message])
    result=model.predict(msg_vec)
    print(f"Message:{message}")
    print(f"Prediction:{result[0].upper()}")
    print()
    
predict("Congratulations! You won a free iPhone. Click here to claim now!")
predict("Hey are you coming to college tomorrow?")
predict("FREE entry WIN cash prizes call now!")
predict(" want free money")
predict("want free money now click here")
predict("you have won free money claim now")
predict("want free money transfer to your account")