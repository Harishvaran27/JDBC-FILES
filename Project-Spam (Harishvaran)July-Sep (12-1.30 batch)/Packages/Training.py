# For manipulating Dataframe:
import pandas as pd 

# Machinecode convertor:
from sklearn.feature_extraction.text import CountVectorizer

# Spliting Train and Test dataset:
from sklearn.model_selection import train_test_split

# Model Creation:
from sklearn.naive_bayes import MultinomialNB

# saving the ML model:
import joblib

def TrainingTheModel():
    # Reading the Spam Dataset:
    df = pd.read_csv('spam_ham_dataset.csv',encoding = "ISO-8859-1")

   
    # Renaming the column names with respective instances:
    

    # Spliting up to Features and Label
    Features = df.text
    Label = df.label
    print(df)


    # Converting the object format to binary/Machine code:
    cv = CountVectorizer()
    Features_cv = cv.fit_transform(Features)

    global X_test,y_test

    # Spliting up to train and test:
    X_train,X_test,y_train,y_test = train_test_split(Features_cv,Label,test_size = 0.3)

    
    # Model creation:
    model = MultinomialNB()

    # Training the model:
    model.fit(X_train,y_train)

    # saving the ML model:
    FileName = 'ML-SVM-Model.sav'
    joblib.dump(model,FileName)
TrainingTheModel()
# print(X_test)
# print(y_test)
