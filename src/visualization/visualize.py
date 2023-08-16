import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

def conf_matrix(y_true,y_pred,path):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true,y_pred)

    fig, ax = plt.subplots(figsize=(5, 5),dpi=100)

    # Create a ConfusionMatrixDisplay object and plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.suptitle('Confusion Matrix')
    plt.savefig(path)


def plot_roc_det(classifiers,X_test,y_test,path):
    """
    Plot ROC, Precision-Recall, DET, and Calibration curves for a set of classifiers.

    Parameters:
    classifiers (dict): Dictionary containing classifier names as keys and classifier (fitted) instances as values.
    X_test (pd.DataFrame): Validation dataset features.
    y_test (pd.Series): Validation dataset target.
    path (File path): System directory.

    Returns:
    None
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs =  axs.flatten()

    for name, clf in classifiers.items():
        preds = clf.predict_proba(X_test)[:,1]

        RocCurveDisplay.from_predictions(y_test, preds, ax=axs[0], name=name)
        PrecisionRecallDisplay.from_predictions(y_test, preds, ax=axs[1], name=name)
        DetCurveDisplay.from_predictions(y_test, preds, ax=axs[2], name=name)
        CalibrationDisplay.from_predictions(y_test, preds, ax=axs[3], name=name)

    axs[0].set_title("ROC curves")
    axs[1].set_title("PR curves")
    axs[2].set_title("DET curves")
    axs[3].set_title("CAL curves")

    plt.legend()
    plt.suptitle('Model Evaluation Curves')
    plt.savefig(path)