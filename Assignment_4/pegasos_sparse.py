from aml_perceptron import LinearClassifier, sparse_dense_dot, add_sparse_to_dense
import random
import numpy as np


class PegasosSVCSparse(LinearClassifier):
    """
    Pegasos algorithm using SVC and hinge loss
    """

    def __init__(self, n_iter=1e5):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """

        self.n_iter = int(n_iter)

        # Just to avoid warnings about using non-defined attributes
        self.regularizationParameter = None
        self.w = None

    def fit(self, X, Y):
        """
        Train the pegasos hinge loss model
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        self.regularizationParameter = 1 / len(self.w)

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        data = tuple(zip(X, Ye))

        allLoss = []

        # Scaling factor for vector scaling speedup
        scalingFactor = 1.0

        # Avoid calling random.choice() every iteration
        randomIndex = tuple([random.randrange(len(data)) for _ in range(self.n_iter)])

        # Pegasos algorithm using hinge loss
        for t in range(1, self.n_iter + 1):
            x, y = data[randomIndex[t - 1]]

            # Eta, learning rate
            learningRate = 1 / (self.regularizationParameter * t)

            # Compute the output score for this instance.
            score = scalingFactor * sparse_dense_dot(x, self.w)

            scalingFactor = (1 - learningRate * self.regularizationParameter) * scalingFactor

            if scalingFactor == 0.:
                scalingFactor = 1e-9

            # Add gradient if `y.(w.x) < 1`
            if y * score < 1:
                add_sparse_to_dense(x, self.w, learningRate * y / scalingFactor)
                allLoss.append(1 - y * score)
            else:
                allLoss.append(0)

            if t != 0:
                if t < 1e4 and t % 1e3 == 0:
                    avg = sum(allLoss) / len(allLoss)
                    print("Iteration {}*10^3, average loss: {:.4f}".format(round(t / 1e3), avg))

                if t < 1e5 and t % 1e4 == 0:
                    avg = sum(allLoss) / len(allLoss)
                    print("Iteration {}*10^4, average loss: {:.4f}".format(round(t / 1e4), avg))

        self.w *= scalingFactor


class PegasosLRSparse(LinearClassifier):
    """
    Pegasos algorithm using LR and logistic loss
    """

    def __init__(self, n_iter=1e5):
        """
        The constructor can optionally take a parameter n_iter specifying many pair to train on.
        """

        self.n_iter = int(n_iter)

        # Just to avoid warnings about using non-defined attributes
        self.regularizationParameter = None
        self.w = None

    def fit(self, X, Y):
        """
        Train the pegasos logistic loss model
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])
        # `X.shape[1]` is the number of features

        self.regularizationParameter = 1 / len(self.w)

        data = tuple(zip(X, Ye))

        allLoss = []

        scalingFactor = 1.0

        # Avoid calling random.choice() every iteration
        randomIndex = tuple([random.randrange(len(data)) for _ in range(self.n_iter)])

        # Pegasos algorithm using logistic loss
        for t in range(1, self.n_iter + 1):
            x, y = data[randomIndex[t - 1]]


            # Eta, learning rate
            learningRate = 1 / (t * self.regularizationParameter)

            d = scalingFactor * sparse_dense_dot(x, self.w)

            scalingFactor = (1 - learningRate * self.regularizationParameter) * scalingFactor

            if scalingFactor == 0.:
                scalingFactor = 1e-9

            allLoss.append(np.log(1 + np.exp(-y * d)))

            # Compute the output score for this instance.
            gradient = (y / (1 + np.exp(y * d))) * x

            add_sparse_to_dense(gradient, self.w, learningRate / scalingFactor)

            if t != 0:
                if t < 1e4 and t % 1e3 == 0:
                    avg = sum(allLoss) / len(allLoss)
                    print("Iteration {}*10^3, average loss: {:.4f}".format(round(t / 1e3), avg))

                if t < 1e5 and t % 1e4 == 0:
                    avg = sum(allLoss) / len(allLoss)
                    print("Iteration {}*10^4, average loss: {:.4f}".format(round(t / 1e4), avg))

        self.w *= scalingFactor
