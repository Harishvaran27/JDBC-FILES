import Training as tr
import Testing as te


if __name__ == "__main__":
    
    print()
    ML_Model = "Naive bayes - Multi NominalNB"
    print(f"Used Model: {ML_Model}")
    print()
    tr.TrainingTheModel()
    te.LoadAndPredict()