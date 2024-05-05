import pickle
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.x_mean = None
        self.x_std = None

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape

        # Normalisasi data
        self.x_mean = np.mean(x_train, axis=0)
        self.x_std = np.std(x_train, axis=0)
        x_train_normalized = (x_train - self.x_mean) / self.x_std

        # Inisialisasi bobot dan bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Prediksi nilai target
            y_pred = np.dot(x_train_normalized, self.weights) + self.bias

            # Menghitung gradien
            dw = (1 / n_samples) * np.dot(x_train_normalized.T, (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Regularisasi (jika diperlukan)
            if self.regularization == 'l2':
                self.weights -= self.learning_rate * 2 * self.lambda_ * self.weights

    def predict(self, x_test):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Please train the model using the fit method.")
        # Pastikan melakukan normalisasi yang sama dengan data pelatihan
        x_test_normalized = (x_test - self.x_mean) / self.x_std
        return np.dot(x_test_normalized, self.weights) + self.bias
