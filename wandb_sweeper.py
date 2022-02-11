import wandb
from wandb.keras import WandbCallback
from classifier import Classifier
from lib import load_mnist
import numpy as np
# Set up your default hyperparameters
# Optimized after basian search
hyperparameter_defaults = {
    "n_out": 7,
    "cov_weight": 1,
    "center_weight": 1,
    "mag_weight": 1,
    "n_layers": 2,
    "hidden_size": 64,
    "activation": "leaky_relu",
    "batch_size": 128,
    "epochs": 10,
}

# Pass your defaults to wandb.init
wandb.init(project="multiclass_bin_classifier", config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

n_out = config["n_out"]
cov_weight = config["cov_weight"]
center_weight = config["center_weight"]
mag_weight = config["mag_weight"]
n_layers = config["n_layers"]
hidden_size = config["hidden_size"]
activation = config["activation"]
batch_size = config["batch_size"]
epochs = config["epochs"]


(X_train, y_train), (X_test, y_test) = load_mnist()

# Split teh testing data into a validation set and a test set
(X_valid, y_valid), (X_test, y_test) = (X_test[:5000], y_test[:5000]), (X_test[5000:], y_test[5000:])

classifier = Classifier(
    X_train, y_train,
    hidden_sizes=[hidden_size] * n_layers,
    n_out=n_out,
    activation=activation,
    n_classes=10,
    cov_weight=cov_weight,
    center_weight=center_weight,
    mag_weight=mag_weight,
)

classifier.train_buckets()
starting_validation_accuracy = classifier.evaluate(X_test, y_test)
wandb.log({"validation_accuracy": starting_validation_accuracy})

# Log metrics inside your training loop
for epoch in range(config["epochs"]):
    classifier.train_nn(epochs=1)
    classifier.train_buckets()
    validation_accuracy = classifier.evaluate(X_valid, y_valid)
    metrics = {"validation_accuracy": validation_accuracy}
    wandb.log(metrics)


outputs = classifier.get_outputs(X_valid) # (batch_size, n_out)
cov = np.mean(np.cov(outputs), axis=0) # (n_out,)
center = np.mean(outputs, axis=0) # (n_out,)
mag = np.mean(np.abs(outputs), axis=0) # (n_out,)
wandb.log({
    "covariance": wandb.Histogram(cov),
    "average": wandb.Histogram(center),
    "magnitude": wandb.Histogram(mag),
})
