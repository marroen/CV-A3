import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the layer before output (FC2)"""
    model.eval()
    embeddings = []
    labels = []
    
    # Hook to capture FC2 output
    def hook(module, input, output):
        embeddings.append(output.detach().cpu().numpy())
    
    handle = model.fc2.register_forward_hook(hook)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            _ = model(images)
            labels.append(targets.numpy())
    
    handle.remove()
    return np.concatenate(embeddings), np.concatenate(labels)

def plot_tsne(embeddings, labels, class_names, model_name):
    """Perform t-SNE dimensionality reduction and plot results"""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("hls", len(class_names))
    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        alpha=0.7
    )
    
    # Create legend with class names
    handles, _ = scatter.get_legend_handles_labels()
    plt.legend(handles, class_names, bbox_to_anchor=(1.05, 1), loc=2)
    
    plt.title(f'Model {model_name} t-SNE Visualization of FC2 Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

def analyze_confusions(embeddings, labels, class_names):
    """Analyze potential class confusions using clustering"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    
    # Cluster embeddings into 10 groups
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Create confusion matrix between true labels and clusters
    cm = confusion_matrix(labels, cluster_labels)
    
    # Find most confused class pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            confusion_score = cm[i,j] + cm[j,i]
            confusion_pairs.append(((i,j), confusion_score))
    
    # Sort by most confused pairs
    confusion_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nMost likely confusion pairs:")
    for (i,j), score in confusion_pairs[:3]:
        print(f"{class_names[i]} ↔ {class_names[j]} (score: {score})")