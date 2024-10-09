from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D

def build_model(input_shape=(224, 224, 3), num_classes=100):
    # Load the VGG16 model, pre-trained on ImageNet data
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(vgg_conv)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model