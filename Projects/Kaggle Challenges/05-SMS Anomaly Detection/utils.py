from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def model_report(y, y_pred):
        
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1:", f1_score(y, y_pred))
    print('\n', '-' * 40 + '\n')
    print("Confusion Matrix: \n", confusion_matrix(y, y_pred))
    print('\n', '-' * 40 + '\n')
    print("Classification Report: \n", classification_report(y, y_pred))