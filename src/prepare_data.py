import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    # Define the data directory and CSV file path
    data_dir = './data'  # Adjust this path if your data directory is different
    csv_file = os.path.join(data_dir, 'variety_classification.csv')
    
    # Load the CSV file
    try:
        labels = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
    except FileNotFoundError:
        print(f"CSV file not found at {csv_file}. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return
    
    # Clean column names
    labels.columns = labels.columns.str.strip().str.lower()
    
    # Rename 'variety_image_path' to 'image_path' for clarity
    if 'variety_image_path' in labels.columns:
        labels.rename(columns={'variety_image_path': 'image_path'}, inplace=True)
    else:
        print("The column 'variety_image_path' is not found in the CSV.")
        return
    
    # Check if 'class' column exists
    if 'class' not in labels.columns:
        print("The column 'class' is not found in the CSV.")
        return
    
    # Use 'class' column as labels
    labels['label'] = labels['class']
    
    # Adjust image paths to include the data directory
    labels['image_path'] = labels['image_path'].apply(lambda x: os.path.join(data_dir, x))
    
    # Verify that the image files exist
    missing_files = labels[~labels['image_path'].apply(os.path.exists)]
    if not missing_files.empty:
        print("Warning: The following image files were not found:")
        print(missing_files['image_path'])
        # Optionally, remove rows with missing files
        labels = labels[labels['image_path'].apply(os.path.exists)]
    
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
    generator = datagen.flow_from_dataframe(
        dataframe=labels,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    print("Data generator set up successfully. Ready for model training.")
    
    # Proceed with model training or other processing...
    # For example:
    # model.fit(generator, epochs=10)

if __name__ == '__main__':
    main()
