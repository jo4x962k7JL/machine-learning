#
import random, os, glob, sys, dlib, cv2, time, keras
from sklearn.datasets import load_files
from extract_bottleneck_features import *
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load files
def load_dataset(path):
    data = load_files(path)
    dog_targets = keras.utils.np_utils.to_categorical(np.array(data['target']), 133)
    return dog_targets

train_targets = load_dataset('dogImages/train')
valid_targets = load_dataset('dogImages/valid')
test_targets = load_dataset('dogImages/test')
dog_names = [item[20:-1] for item in sorted(glob.glob('dogImages/train/*/'))]

# Human detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces1 = detector(gray, 0)
    faces2 = face_cascade.detectMultiScale(gray)
    if len(faces1) > 0 and len(faces2) > 0: return True
    return False

# Dog detector(using ResNet-50 pretrained on ImageNet)
ResNet50_model = keras.applications.resnet50.ResNet50(weights='imagenet')
def path_to_tensor(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)
def dog_detector(img_path):
    img = keras.applications.resnet50.preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

print('\n======================== Create a CNN to Classify Dog Breeds (using Transfer Learning) ========================\n')
start = time.time()
bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']

# valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']
Xception_model = keras.models.Sequential()
Xception_model.add(keras.layers.GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(keras.layers.Dense(133, activation = 'softmax'))
Xception_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=8e-5), metrics=['accuracy'])
# checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', verbose=False, save_best_only=True)
# hist_Xception = Xception_model.fit(train_Xception, train_targets, validation_data=(valid_Xception, valid_targets),
          # epochs=30, batch_size=50, callbacks=[checkpointer], verbose=False, shuffle=True)
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]
test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: {:.4f}%'.format(test_accuracy))
print('Run time: {:.2f}seconds'.format((time.time() - start)))

# For fun
def Xception_predict_breed(img_path):
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]
def funfact(img_path):
    dog_breed = Xception_predict_breed(img_path)
    if face_detector(img_path):
        print('\n\nHi Human! Your dog breed is {}\n'.format(dog_breed))
    elif dog_detector(img_path):
        print('\n\nHi Dog! I guess your breed is {}\n'.format(dog_breed))
    else:
        print('\n\nYou are neither human nor dog!\n')
print(funfact(sys.argv[1]))
