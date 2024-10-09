import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_generator(data_dir, segmentation=False):
    labels_csv = os.path.join(data_dir, 'variety_classification.csv')
    
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"The labels file was not found at the path: {labels_csv}")
    
    # Read the CSV file
    labels = pd.read_csv(labels_csv, low_memory=False)
    
    # Check if the 'id' column exists but is empty, indicating misalignment
    if 'id' in labels.columns and labels['id'].isnull().all():
        # Shift columns to align data correctly
        labels = labels.shift(-1, axis='columns')
        labels = labels.iloc[:, :-1]  # Remove the last column which is all NaN after shift
        # Rename columns to match data
        labels.columns = ['variety_image_path', 'species', 'variety', 'layout_id', 'for_cropping', 'packed',
                          'amount', 'weight', 'uniform_background', 'spoiled', 'cam', 'city', 'shop', 'crowd',
                          'date', 'subset', 'simp_amount', 'class']
    else:
        # If 'id' column is not empty or does not exist, proceed without shifting
        pass  # Assuming data is correctly aligned
    
    # Clean column names
    labels.columns = labels.columns.str.strip().str.lower()
    
    # Ensure required columns exist
    required_columns = ['variety_image_path', 'class']
    for col in required_columns:
        if col not in labels.columns:
            raise ValueError(f"CSV file must contain '{col}' column.")
    
    # Create 'filename' column by joining data_dir and 'variety_image_path'
    labels['filename'] = labels['variety_image_path'].apply(lambda x: os.path.join(data_dir, x.strip()))
    
    # Use 'class' column as labels
    labels['label'] = labels['class'].astype(str).str.strip()
    
    # Verify that the image files exist
    missing_files = labels[~labels['filename'].apply(os.path.exists)]
    if not missing_files.empty:
        print("Warning: The following image files were not found:")
        print(missing_files['filename'])
        # Remove rows with missing files to prevent errors
        labels = labels[labels['filename'].apply(os.path.exists)]
    
    if labels.empty:
        raise ValueError("No valid image files found. Please check your image paths.")
    
    # Set up the image data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Create the data generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=labels,
        x_col='filename',
        y_col='label',
        directory=None,  # 'filename' contains full paths
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32
    )
    
    return train_generator

if __name__ == '__main__':
    data_dir = './data'  # Adjust this path if necessary
    try:
        train_generator = setup_generator(data_dir)
        print("\nGenerator setup successful. Ready to train model.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
