#
# import matplotlib
# matplotlib.use('Agg')
import random, os, glob, tqdm, dlib, cv2, time
from keras.utils import np_utils, generic_utils
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras import backend as K
from scipy import misc, ndimage
from sklearn.datasets import load_files
from PIL import ImageFile
from urllib.request import urlretrieve
from extract_bottleneck_features import *
import numpy as np
import matplotlib.pyplot as plt
random.seed(8675309)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load files
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')
dog_names = [item[20:-1] for item in sorted(glob.glob('dogImages/train/*/'))]
print('There are {} total dog categories.'.format(len(dog_names)))
print('There are {} total dog images.'.format(len(np.hstack([train_files, valid_files, test_files]))))
print('-'*50)
print('There are {} training dog images.'.format(len(train_files)))
print('There are {} validation dog images.'.format(len(valid_files)))
print('There are {} test dog images.'.format(len(test_files)))
human_files = np.array(glob.glob('lfw/*/*'))
random.shuffle(human_files)
print('There are {} total human images.'.format(len(human_files)))

# Human detector
start = time.time()
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def percentage_predicted(img_file, detector):
    return 100.0*([detector(i) for i in img_file].count(True))/len(img_file)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces1 = detector(gray, 0)
    faces2 = face_cascade.detectMultiScale(gray)
    if len(faces1) > 0 and len(faces2) > 0: return True
    return False

print('\nUsing Human face detector:')
print('{:.2f}% of {} HUMAN images(in human_files) have a detected HUMAN face.'.format(percentage_predicted(human_files, face_detector), len(human_files)))
print('{:.2f}% of {} DOG images(in train_files) have a detected HUMAN face.'.format(percentage_predicted(train_files, face_detector), len(train_files)))
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

# Dog detector(using ResNet-50 pretrained on ImageNet)
start = time.time()
ResNet50_model = ResNet50(weights='imagenet')
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)
def dog_detector(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

print('\nUsing Dog detector:')
print('{:.2f}% of {} HUMAN images(in human_files) have a detected DOG.'.format(percentage_predicted(human_files, dog_detector), len(human_files)))
print('{:.2f}% of {} DOG images(in train_files) have a detected DOG.'.format(percentage_predicted(train_files, dog_detector), len(train_files)))
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

# CNN Model plot
def save_plot_model(model_details, filename):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
    axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(filename)

# Classify Dog Breeds (from Scratch)
print('\n======================== Create a CNN to Classify Dog Breeds (from Scratch) ========================\n')
start = time.time()
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# Model Architecture
def swish(x):
    return (K.sigmoid(x) * x)
generic_utils.get_custom_objects().update({'swish': swish})

model = Sequential()
model.add(Conv2D(32, kernel_size=(5), activation='swish', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.75, epsilon=0.001))
model.add(Conv2D(64, kernel_size=(4), activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.75, epsilon=0.001))
model.add(Conv2D(128, kernel_size=(3), activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.75, epsilon=0.001))
model.add(Conv2D(256, kernel_size=(2), activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.75, epsilon=0.001))
model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))
print(model.summary())
# Compile the Model
model.compile(optimizer=Adam(lr=2e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the Model
epochs = 20
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch_withdropout.hdf5', verbose=1, save_best_only=True)
hist = model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
# Load the Model with the Best Validation Loss
model.load_weights('saved_models/weights.best.from_scratch_withdropout.hdf5')
# Test the Model
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: {:.4f}%'.format(test_accuracy))
# Plot the model
save_plot_model(hist, 'dog_breed_classifier_from_scratch.png')
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

# Create a CNN to Classify Dog Breeds (using Transfer Learning)
print('\n======================== Create a CNN to Classify Dog Breeds (using Transfer Learning) ========================\n')
start = time.time()
bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(133, activation = 'softmax'))
print(Xception_model.summary())
Xception_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=8e-5), metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', verbose=1, save_best_only=True)
hist_Xception = Xception_model.fit(train_Xception, train_targets, validation_data=(valid_Xception, valid_targets),
          epochs=30, batch_size=50, callbacks=[checkpointer], verbose=1, shuffle=True)
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: {:.4f}%'.format(test_accuracy))
save_plot_model(hist_Xception, 'dog_breed_classifier_using_transfer_learning.png')
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

# For fun
start = time.time()
def Xception_predict_breed(img_path):
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def save_plot_results(file_paths):
    for idx, image in enumerate(file_paths):
        dog_breed = Xception_predict_breed(image)
        for root, dirs, files in os.walk('dogImages/train/'):
            if dog_breed in root:
                family = root + '/' + files[0]
                break
        fig = plt.figure(figsize=(30,30))
        k = 1
        fig.set_size_inches(4, 2)
        fig.add_subplot(1, 2, k)
        plt.imshow(np.expand_dims(ndimage.imread(image), 0)[0])
        if face_detector(image):
            plt.title('Hi Human!\nYour dog breed is {}'.format(dog_breed), fontdict={'fontsize':8})
            plt.axis('off')
            k += 1
            fig.add_subplot(1, 2, k)
            plt.imshow(np.expand_dims(ndimage.imread(family), 0)[0])
            plt.title('This is your family!', fontdict={'fontsize':8})
            plt.axis('off')
            plt.savefig('funfact'+str(idx)+'.jpg',dpi=200)
        elif dog_detector(image):
            plt.title('Hi Dog!\nI guess your breed is {}'.format(dog_breed), fontdict={'fontsize':8})
            plt.axis('off')
            k += 1
            fig.add_subplot(1, 2, k)
            plt.imshow(np.expand_dims(ndimage.imread(family), 0)[0])
            plt.title('This is your family!', fontdict={'fontsize':8})
            plt.axis('off')
            plt.savefig('funfact'+str(idx)+'.jpg', dpi=200)
        else:
            plt.title('You are neither human nor dog!', fontdict={'fontsize':8})
            plt.axis('off')
            plt.savefig('funfact'+str(idx)+'.jpg', dpi=200)

file_paths = ['test_images/Boxer.jpg',
              'test_images/Labrador_retriever.jpg',
              'test_images/Pomeranian.jpg',
              'test_images/Golden_retriever.jpg',
              'test_images/woman.jpg',
              'test_images/man.jpg']
save_plot_results(file_paths, 'fun_fact')
print('Run time: {:.2f}mins'.format((time.time() - start)/60))
