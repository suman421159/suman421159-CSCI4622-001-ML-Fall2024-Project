import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_data(data_dir, segmentation=False):
    if segmentation:
        train_dir = os.path.join(data_dir, 'segmentation_train', 'train')
        labels_csv = os.path.join(data_dir, 'variety_classification.csv')
    else:
        train_dir = os.path.join(data_dir, 'train')
        labels_csv = os.path.join(data_dir, 'variety_classification.csv')
    
    labels = pd.read_csv(labels_csv)
    labels['path'] = labels['id'].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
    return train_dir, labels

def visualize_data(train_dir, labels):
    sns.countplot(x='variety', data=labels)
    plt.title('Distribution of Varieties')
    plt.xticks(rotation=90)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, variety in enumerate(labels['variety'].unique()[:2]):
        sample_file = labels[labels['variety'] == variety].iloc[0]['path']
        img = Image.open(sample_file)
        axes[i].imshow(img)
        axes[i].set_title(f'Sample Image: Variety {variety}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_dir = './data'
    train_dir, labels = load_data(data_dir)
    visualize_data(train_dir, labels)