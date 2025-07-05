# Logistic regression based model/algorithm
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

def initialize_matrices(X):
    w = np.zeros((X.shape[0], 1))
    b = float(0)

    return w, b

def sigmoid(Z):
    return (1 / (1 + (np.exp(-Z))))

def calculate_logistic_regression(X, w, b):
    return sigmoid(np.dot(w.T, X) + b)

def calculate_vectorized_loss(Y, predicted_Y):
    return (Y * np.log(predicted_Y) + (1 - Y) * np.log(1 - predicted_Y))

def train(X, Y, w, b, learning_rate=0.0005, num_iterations = 10000):
    m = X.shape[1]

    for it in range(num_iterations):
        A = calculate_logistic_regression(X, w, b)
        cost = 1 - (- np.sum(calculate_vectorized_loss(Y.T, A)) / m) 

        Z = (A - Y.T)
        dw = (np.dot(X, Z.T)) / m
        db = np.sum(Z) / m

        w -= learning_rate * dw
        b -= learning_rate * db

        print(f'Cost after iteration {it} -> {round(cost * 100, 2)}')
    
    return w, b

def predict(X, w, b, labels):
    prediction_set = calculate_logistic_regression(X, w, b).T

    total_examples = len(prediction_set)
    total_correct_prediction = 0

    for i, _ in enumerate(prediction_set):
        prediction = 'cat' if prediction_set[i] > 0.5 else 'dog'
        label = "cat" if labels[i] == 1 else "dog"
        print(f'Prediction for example {i} -> {prediction} | correct label -> {label}')
        if prediction == label:
            total_correct_prediction += 1
    
    print(f'Prediction result: {total_correct_prediction} predicted correctly over {total_examples} examples')
    print(f'With an accuracy of {round(total_correct_prediction / total_examples * 100, 2)}%')



def load_imgs(load_train_set=True, image_limit=2000):
    image_set = 'training_set' if load_train_set else 'test_set'
    folder_path = f'./{image_set}'
    images = []
    labels = []

    # load cat images
    loaded_image_index = 0
    cat_folder_path = folder_path + '/cats'
    for file in os.listdir(cat_folder_path):
        if loaded_image_index > image_limit:
            break;
        img = cv2.imread(os.path.join(cat_folder_path, file))
        if img is not None:
            img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            labels.append(1)
            loaded_image_index += 1

    # load dog images
    loaded_image_index = 0
    dog_folder_path = folder_path + '/dogs'
    for file in os.listdir(dog_folder_path):
        if loaded_image_index > image_limit:
            break;
        img = cv2.imread(os.path.join(dog_folder_path, file))
        if img is not None:
            img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            labels.append(0)
            loaded_image_index += 1

    np_images = np.stack(images, axis=0)
    np_images = np_images.reshape(np_images.shape[0], -1).T / 255
    np_labels = np.array(labels)
    np_labels = np_labels.reshape(np_labels.shape[0], 1)
    return np_images, np_labels



if __name__ == "__main__":
    # train the model
    print('Process started...')
    train_set_images, labels = load_imgs()
    w, b = initialize_matrices(train_set_images)
    w, b = train(train_set_images, labels, w, b)
    test_set_images, labels = load_imgs(load_train_set=False)
    predict(test_set_images, w, b, labels)
    print('Process finished...')