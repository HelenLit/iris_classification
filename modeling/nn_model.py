import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
from utils import setup_logger

logger = setup_logger(__name__)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(Model, self).__init__()
        try:
            self.layer1 = nn.Linear(input_dim, 64)
            self.layer2 = nn.Linear(64, 16)
            self.drop = nn.Dropout(dropout_rate)
            self.layer3 = nn.Linear(16, output_dim)
            logger.info(f"Model initialized successfully:\n{str(self)}")
        except Exception as e:
            logger.error(f"Error initializing the model: {e}")
            raise

    def forward(self, x):
        try:
            x = F.relu(self.layer1(x))
            x = self.drop(self.layer2(x))
            x = F.relu(x)
            x = self.layer3(x)
            return x
        except Exception as e:
            logger.error(f"Error during the forward pass: {e}")
            raise

    def fit(self, X_tr_tensor, y_tr_tensor, epochs=400, learning_rate=0.002, plot=False):
        try:
            logger.info("Starting training process...")
            logger.info(f"Set up fit parameters: epochs={epochs}, learning_rate={learning_rate}")

            loss_arr = []
            loss_fn = nn.CrossEntropyLoss()
            optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                try:
                    y_pred = self(X_tr_tensor)
                    loss = loss_fn(y_pred, y_tr_tensor)
                    loss_arr.append(loss.item())

                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:  # Log at the first epoch, every 50 epochs, and at the last epoch
                        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
                except Exception as e:
                    logger.error(f"Error during training at epoch {epoch}: {e}")
                    raise

            if plot:
                try:
                    plt.plot(loss_arr)
                    plt.title("Training Loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.show()
                    logger.info("Training loss plot displayed successfully.")
                except Exception as e:
                    logger.error(f"Error while plotting the training loss: {e}")
                    raise

            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training process: {e}")
            raise

    def predict(self, X_tensor):
        try:
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                logger.info("Generating predictions...")
                y_pred = self(X_tensor)
                predictions = torch.argmax(y_pred, dim=1)  # Get class predictions
                logger.info("Predictions generated successfully.")
                return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, X_tensor, y_true_tensor, metric="f1"):
        try:
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

            logger.info(f"Evaluating the model using {metric} metric...")

            # Generate predictions
            y_pred = self.predict(X_tensor)
            y_true = y_true_tensor.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            # Compute the desired metric
            if metric == "f1":
                score = f1_score(y_true, y_pred, average="weighted")
            elif metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, average="weighted")
            elif metric == "recall":
                score = recall_score(y_true, y_pred, average="weighted")
            else:
                raise ValueError(
                    f"Unsupported metric: {metric}. Choose from 'f1', 'accuracy', 'precision', 'recall'.")

            logger.info(f"Model evaluation completed. {metric.capitalize()} Score: {score:.4f}")
            return score

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def save(self, path: str = None) -> None:
        try:
            logger.info("Saving the model...")
            if not path:
                path = os.path.join(datetime.now().strftime("%Y%m%d_%H%M%S") + '.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
            raise
