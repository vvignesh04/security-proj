# run_federated.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

from client import FLClient
from server import FLServer
from aggregator import secure_average
from encryptionfhe_wrapper import FHEHandler
from models.mental_health_model import SimpleModel


def get_mental_health_loader():
    df = pd.read_csv("D:/s6/security/proj/data/synthetic_phq9_dataset.csv")
    X = df.drop("label", axis=1).values.astype("float32")
    y = df["label"].values.astype("int64")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=10, shuffle=True)


if __name__ == '__main__':
    global_model = SimpleModel()
    server = FLServer(global_model)
    fhe = FHEHandler()
    clients = [FLClient(SimpleModel(), get_mental_health_loader()) for _ in range(3)]

    for round in range(1, 6):
        print(f"\n--- Federated Round {round} ---")
        encrypted_updates = []

        for client in clients:
            local_weights = client.train(epochs=1)
            encrypted_weights = fhe.encrypt(local_weights)
            encrypted_updates.append(encrypted_weights)

        avg_encrypted = secure_average(encrypted_updates)
        decrypted_avg = fhe.decrypt(avg_encrypted, global_model.state_dict())
        global_model.load_state_dict(decrypted_avg)

    print("\nFederated training complete.")

    # Evaluate global model
    global_model.eval()
    correct = 0
    total = 0
    test_loader = get_mental_health_loader()

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test data: {100 * correct / total:.2f}%")

    # Save the final global model
    torch.save(global_model.state_dict(), "D:\s6\security\proj\data/global_model.pth")
