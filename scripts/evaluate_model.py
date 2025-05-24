import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mental_health_model import SimpleModel

def get_mental_health_loader(path="data/synthetic_phq9_dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1).values.astype("float32")
    y = df["label"].values.astype("int64")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=10, shuffle=False)

def evaluate(model_path="saved_models/global_model.pth", data_path="data/synthetic_phq9_dataset.csv"):
    model = SimpleModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = get_mental_health_loader(data_path)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test data: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate()
