import matplotlib.pyplot as plt
import pandas as pd
import os

# Create plots directory if it doesn't exist
os.makedirs('training_plots', exist_ok=True)

# List of model names
model_names = ['model_a', 'model_b', 'model_c']

# Create plots for each model
for model_name in model_names:
    # Read CSV file
    csv_path = f"{model_name}_metrics.csv"
    df = pd.read_csv(csv_path)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], 'r-', label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], 'b-', label='Validation Loss')
    plt.title(f'{model_name.upper()} Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_accuracy'], 'r-', label='Training Accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], 'b-', label='Validation Accuracy')
    plt.title(f'{model_name.upper()} Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'training_plots/{model_name}_metrics.png')
    plt.close()
    
    print(f'Generated plots for {model_name} in training_plots/ directory')