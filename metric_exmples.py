import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
)

classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
classifier.fit(X_train, y_train)

display = RocCurveDisplay.from_estimator(
    classifier, X_test, y_test, name="Example Classifier", plot_chance_level=True
)


display.plot(color='orange')
display.ax_.set_title("2-class ROC curve")




#save plot
plt.savefig(f'roc_plot_example.png')
