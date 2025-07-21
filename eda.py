import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------- 1. Load CIFAR Dataset ----------
def load_cifar(dataset="cifar10", train=True):
    transform = transforms.ToTensor()
    if dataset == "cifar10":
        data = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
        classes = data.classes
    elif dataset == "cifar100":
        data = torchvision.datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)
        classes = data.classes
    else:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")
    return data, classes


# ---------- 2. Class Distribution ----------
def class_distribution(data, classes, title="Class Distribution"):
    counts = defaultdict(int)
    for _, label in data:
        counts[label] += 1

    total = sum(counts.values())
    percentages = [counts[i] / total * 100 for i in range(len(classes))]

    plt.figure(figsize=(10, 5))
    plt.bar(classes, percentages)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.show()


# ---------- 3. Color Distribution per Class ----------
def color_distribution(data, classes):
    color_sums = {cls: [0, 0, 0] for cls in range(len(classes))}
    counts = defaultdict(int)

    for img, label in data:
        color_sums[label][0] += img[0].mean().item()
        color_sums[label][1] += img[1].mean().item()
        color_sums[label][2] += img[2].mean().item()
        counts[label] += 1

    r_means = [color_sums[i][0] / counts[i] for i in range(len(classes))]
    g_means = [color_sums[i][1] / counts[i] for i in range(len(classes))]
    b_means = [color_sums[i][2] / counts[i] for i in range(len(classes))]

    x = np.arange(len(classes))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, r_means, 0.2, color="red", label="Red")
    plt.bar(x, g_means, 0.2, color="green", label="Green")
    plt.bar(x + 0.2, b_means, 0.2, color= "blue", label="Blue")

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Mean Channel Intensity")
    plt.title("Mean RGB per Class")
    plt.legend()
    plt.show()


# ---------- 4. Mean & Variance per Class ----------
def mean_variance_per_class(data, classes):
    means, variances = [], []
    for i in range(len(classes)):
        cls_pixels = [img.mean().item() for img, label in data if label == i]
        means.append(np.mean(cls_pixels))
        variances.append(np.var(cls_pixels))

    x = np.arange(len(classes))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.15, means, 0.3, label="Mean")
    plt.bar(x + 0.15, variances, 0.3, label="Variance")
    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Value")
    plt.title("Mean & Variance of Pixel Intensity per Class")
    plt.legend()
    plt.show()


# ---------- 5. NEW: Mean Intensity per Class (Bar Chart) ----------
def mean_intensity_per_class(data, classes):
    """
    Mean Intensity = Average of (R+G+B)/3 for all pixels in all images of a class
    """
    intensities = []
    for i in range(len(classes)):
        cls_vals = []
        for img, label in data:
            if label == i:
                # Average intensity for each image
                gray_vals = img.mean(dim=0).mean().item()  # (R+G+B)/3 averaged over image
                cls_vals.append(gray_vals)
        intensities.append(np.mean(cls_vals))

    plt.figure(figsize=(12, 6))
    plt.bar(classes, intensities)
    plt.xticks(rotation=45)
    plt.ylabel("Mean Intensity")
    plt.title("Average Pixel Intensity per Class")
    plt.show()


# ---------- 6. Run EDA ----------
if __name__ == "__main__":
    dataset_name = "cifar100"  # change to "cifar100" if needed
    train_data, class_names = load_cifar(dataset_name, train=True)

    class_distribution(train_data, class_names, f"{dataset_name.upper()} Class Distribution")
    color_distribution(train_data, class_names)
    mean_variance_per_class(train_data, class_names)
    mean_intensity_per_class(train_data, class_names)

