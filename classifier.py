import numpy as np
from lib import multi_binary_mlp, multiclass_loss

class Classifier:

    def __init__(self, X, y, hidden_sizes, n_out, activation="relu", n_classes=10, cov_weight=1, center_weight=1, mag_weight=1):
        self.model = multi_binary_mlp(hidden_sizes=hidden_sizes, n_out=n_out, activation=activation)
        loss_fn = multiclass_loss(cov_weight=cov_weight, center_weight=center_weight, mag_weight=mag_weight)
        self.model.compile(optimizer="adam", loss=loss_fn)

        self.n_out = n_out
        self.X = X
        self.y = y

        self.buckets = None
        self.n_classes = n_classes
    
    
    def train_nn(self, epochs=10, batch_size=128):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size)

    def get_outputs(self, X):
        return self.model.predict(X) # (batch_size, n_out)

    def outputs_to_index(self, outputs):
        bin_outputs = (outputs > 0).astype(int) # (batch_size, n_out)

        # interpret bin_outputs as a batch of binary numbers
        # convert each binary number to a decimal number
        two_powers = 2**np.arange(outputs.shape[1]) # (1, n_out)
        indecies = bin_outputs.dot(two_powers) # (batch_size,)
        return indecies


    def train_buckets(self):
        outputs = self.get_outputs(self.X) # (batch_size, n_out)
        indecies = self.outputs_to_index(outputs)
        n_buckets = 2**self.n_out
        self.buckets = np.zeros((n_buckets, self.n_classes)) # (n_buckets, n_classes)
        for i, index in enumerate(indecies):
            self.buckets[index, self.y[i]] += 1
        
        # normalize each row
        self.buckets = self.buckets / (self.buckets.sum(axis=1, keepdims=True) + 1e-8)


    def predict(self, X):
        if self.buckets is None:
            # raise an error
            raise Exception("You must train the buckets before you can predict: classifier.train_buckets()")
        
        outputs = self.get_outputs(X) # (batch_size, n_out)
        indecies = self.outputs_to_index(outputs)
        return self.buckets[indecies]
    

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.argmax(axis=1)
        return np.mean(y_pred == y)