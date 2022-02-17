from aml_perceptron import LinearClassifier
import random
import numpy as np


class PegasosHingeLoss(LinearClassifier):
    """
    Pegasos algorithm using hinge loss
    """

    def __init__(self, n_iter=1e4):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """

        self.n_iter = int(n_iter)

        # Just to avoid warnings about using non-defined attributes
        self.regularizationParameter = None
        self.w = None

    def fit(self, X, Y, lam=1e-3):
        """
        Train the pegasos hinge loss model
        """

        self.regularizationParameter = lam

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]

        # Weights
        self.w = np.zeros(n_features)

        # Features and output
        data = list(zip(X, Ye))

        # Pegasos algorithm using hinge loss
        for t in range(1, self.n_iter + 1):
            # Training pair
            x, y = random.choice(data)

            # Eta, learning rate
            learningRate = 1 / (self.regularizationParameter * t)

            # Compute the output score for this instance.
            score = x.dot(self.w)

            # If there was an error, update the weights.
            self.w = (1 - learningRate * self.regularizationParameter) * self.w

            # Add gradient if `y.(w.x) < 1`
            if y * score < 1:
                self.w += (learningRate * y) * x


class PegasosLogisticLoss(LinearClassifier):
    """
    Pegasos algorithm using logistic loss
    """

    def __init__(self, n_iter=1e4):
        """
        The constructor can optionally take a parameter n_iter specifying many pair to train on.
        """

        self.n_iter = int(n_iter)

        # Just to avoid warnings about using non-defined attributes
        self.regularizationParameter = None
        self.w = None

    def fit(self, X, Y, lam=1e-3):
        """
        Train the pegasos logistic loss model
        """

        self.regularizationParameter = lam

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])
        # `X.shape[1]` is the number of features

        data = list(zip(X, Ye))

        # Pegasos algorithm using logistic loss
        for t in range(1, self.n_iter + 1):
            x, y = random.choice(data)

            # Eta, learning rate
            learningRate = 1 / (t * self.regularizationParameter)

            # Compute the output score for this instance.
            gradient = (y / (1 + np.exp(y * x.dot(self.w)))) * x
            self.w = (1 - learningRate * self.regularizationParameter) * self.w + learningRate * gradient
