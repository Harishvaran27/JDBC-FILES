# Loading the model:
import joblib

# Loading the test dataset:
from Training import TrainingTheModel
from Training import X_test,y_test
# Testing the Model:
# TrainingTheModel()
from sklearn.metrics import accuracy_score,confusion_matrix
def LoadAndPredict():

    # Loading the Model:
    model = joblib.load(filename="ML-SVM-Model.sav")


    # Model prediction:
    y_act = model.predict(X_test)


    # accuracy of the model:
    print(f"Model accuracy :{accuracy_score(y_act,y_test)}")
    print()

    # confusion matrix:
    print("Confusion matrix:")
    print(confusion_matrix(y_act,y_test))
    print()
    # print(y_act)
