import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model_dir = './models'
test_dir = './data/test'
models = [load_model(os.path.join(model_dir, f'visionquest_model_fold_{i}.h5')) for i in range(1, 6)]
datagen = ImageDataGenerator(rescale=1./255)

image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
df_test = pd.DataFrame({
    'filename': image_files
})

print(f"Total images found: {len(image_files)}")
print("Sample image names:", image_files[:5])

test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=test_dir,
    x_col='filename',
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

predictions = [model.predict(test_generator, verbose=1) for model in models]
average_predictions = np.mean(predictions, axis=0).flatten()

submission_df = pd.DataFrame({
    'id': [os.path.basename(path).split('.')[0] for path in test_generator.filenames],  
    'label': average_predictions
})

submission_df['label'] = submission_df['label'].apply(lambda x: format(x, '.3f'))

submission_df.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")