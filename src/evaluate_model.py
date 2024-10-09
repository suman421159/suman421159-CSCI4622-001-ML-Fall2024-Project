import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model_dir = './models'
data_dir = './data'
train_dir = os.path.join(data_dir, 'train')

models = [load_model(os.path.join(model_dir, f'visionquest_model_fold_{i}.h5')) for i in range(1, 6)]

datagen = ImageDataGenerator(rescale=1./255)

labels_csv = os.path.join(data_dir, 'variety_classification.csv')
labels = pd.read_csv(labels_csv)
labels['id'] = labels['id'].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))

test_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,
    x_col='id',
    y_col='variety',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

predictions = [model.predict(test_generator, verbose=1) for model in models]
average_predictions = np.mean(predictions, axis=0)

true_labels = labels['variety'].values

fpr, tpr, _ = roc_curve(true_labels.astype(int), average_predictions.argmax(axis=1))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

np.savetxt("average_predictions.csv", average_predictions, delimiter=",")
print("Average predictions and ROC curve have been processed.")