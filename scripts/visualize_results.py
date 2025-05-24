# scripts/visualize_results.py
import matplotlib.pyplot as plt

rounds = list(range(1, 6))
accuracies = [70, 75, 80, 84, 88]  # Dummy data; log real metrics in future

plt.plot(rounds, accuracies, marker='o')
plt.title("Federated Learning Accuracy Over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("fl_accuracy_curve.png")
plt.show()
