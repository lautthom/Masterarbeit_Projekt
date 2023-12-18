import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def random_forest_model(feature_vectors_train, labels_train, feature_vectors_test, labels_test, show_confusion_matrix=False):
    forest_classifier = RandomForestClassifier(max_depth=20)
    forest_classifier.fit(feature_vectors_train, labels_train)
    predictions = forest_classifier.predict(feature_vectors_test)
    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(labels_test, predictions)
        plt.show()
    return predictions, forest_classifier

def run_loaded_forest(forest_classifier, feature_vectors_test, labels_test, show_confusion_matrix=False):
    predictions = forest_classifier.predict(feature_vectors_test)
    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(labels_test, predictions)
        plt.show()
    return predictions
