import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from scipy.stats import sem
from sklearn import metrics
import cv2
import numpy as np
from scipy.ndimage import zoom
from sklearn import datasets
from matplotlib import pyplot as plt
import os

print("\n\n Please Wait . . . . .\n\n")
num_classes = 2
true_folders = ['../datasets/test_folder/1', '../datasets/train_folder/1']
false_folders = ['../datasets/test_folder/0', '../datasets/train_folder/0']

image_size = 64
pixel_depth = 255.0
image_depth = 3
train_size = 2400
valid_size = 600
test_size = 600

faces = datasets.fetch_olivetti_faces()


# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.target = faces.target
        self.index = 0

    def reset(self):
        print("============================================")
        print("Resetting Dataset & Previous Results.. Done!")
        print("============================================")
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                # print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        print("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
        self.results[str(self.index)] = smile


# Trained classifier's performance evaluation
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Scores: ", (scores))
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# Confusion Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))


# ===============================================================================
# from FaceDetectPredict.py
# ===============================================================================

def detectFaces(frame):
    cascPath = "data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, detected_faces


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                               64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


def predict_face_is_smiling(extracted_face):
    return True if svc_1.predict(extracted_face.reshape(1, -1)) else False


gray1, face1 = detectFaces(cv2.imread("data/Test1.jpg"))
gray2, face2 = detectFaces(cv2.imread("data/Test2.jpg"))
gray3, face3 = detectFaces(cv2.imread("data/Test3.jpg"))
# gray4, face4 = detectFaces(cv2.imread("data/Test4.jpg"))
# gray5, face5 = detectFaces(cv2.imread("data/Test5.jpg"))


def test_recognition(c1, c2):
    extracted_face1 = extract_face_features(gray1, face1[0], (c1, c2))
    print(predict_face_is_smiling(extracted_face1))
    extracted_face2 = extract_face_features(gray2, face2[0], (c1, c2))
    print(predict_face_is_smiling(extracted_face2))
    extracted_face3 = extract_face_features(gray3, face3[0], (c1, c2))
    print(predict_face_is_smiling(extracted_face1))
    # extracted_face4 = extract_face_features(gray4, face4[0], (c1, c2))
    # print(predict_face_is_smiling(extracted_face1))
    # extracted_face5 = extract_face_features(gray5, face5[0], (c1, c2))
    # print(predict_face_is_smiling(extracted_face1))
    cv2.namedWindow("gray1", 0)
    cv2.namedWindow("gray2", 0)
    cv2.namedWindow("gray3", 0)
    cv2.resizeWindow("gray1", 128, 128)
    cv2.resizeWindow("gray2", 128, 128)
    cv2.resizeWindow("gray3", 128, 128)

    cv2.imshow('gray1', extracted_face1)
    cv2.imshow('gray2', extracted_face2)
    cv2.imshow('gray3', extracted_face3)
    # cv2.imshow('gray4', extracted_face4)
    # cv2.imshow('gray5', extracted_face5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(folder):
    """Load the image for a single smile/non-smile lable."""
    for i in range(2):

        image_files = os.listdir(folder[i])
        dataset = np.ndarray(shape=(len(image_files), image_size * image_size),
                             dtype=np.float32)

        image_index = 0
        for image in os.listdir(folder[i]):
            image_file = os.path.join(folder[i], image)
            image_data = (cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(float) -
                          pixel_depth / 2) / pixel_depth

            dataset[image_index] = image_data.flatten()
            image_index += 1
        return dataset


def labelling(true_folders, false_folders):
    data_1 = load_image(true_folders)
    data_0 = load_image(false_folders)
    labels = []
    for item in data_1:
        labels.append(1)
    for item in data_0:
        labels.append(0)
    data = np.vstack((data_1, data_0))

    return data, labels


# ------------------- LIVE FACE RECOGNITION -----------------------------------


if __name__ == "__main__":

    svc_1 = SVC(kernel='poly',
                # gamma=0.015,
                C=0.1, degree=5)  # Initializing Classifier
    # data, target = labelling(true_folders, false_folders)
    trainer = Trainer()
    results = json.load(open("results/results.xml"))  # Loading the classification result
    trainer.results = results

    indices = [int(i) for i in trainer.results]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.results[i] for i in trainer.results]  # Target Vector
    target = np.array(target).astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
    # print(X_test, X_train, y_train, y_test)
    print(cross_val_score(svc_1, X_train, y_train, cv=5))
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
    '''
    param_grid = [
        {'C': [0.1, 1, 10, 100, 1000]}
    ]
    clf = GridSearchCV(svc_1, param_grid, cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_, clf.best_estimator_, clf.best_score_)
    '''
    video_capture = cv2.VideoCapture(0)
    '''
    param_range = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
    train_loss, test_loss = validation_curve(
        SVC(kernel='rbf'), X_train, y_train, param_name='gamma', param_range=param_range, cv=5, scoring='neg_mean_squared_error')

    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    # 可视化图形
    plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

    plt.xlabel("gamma")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

    print(data, target)
    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
    print(cross_val_score(svc_1, X_train, y_train, cv=5))

    # Confusion Matrix and Results
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)



    # 使用validation_curve快速找出参数对模型的影响

    '''

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detectFaces(frame)

        face_index = 0

        cv2.putText(frame, "Press Esc to QUIT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # extract features
                extracted_face = extract_face_features(gray, face, (0.3, 0.05))  # (0.075, 0.05)

                # predict smile
                prediction_result = predict_face_is_smiling(extracted_face)

                # draw extracted face in the top right corner
                frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255,
                                                                                        cv2.COLOR_GRAY2RGB)

                # annotate main image with a label
                if prediction_result is True:
                    cv2.putText(frame, "SMILING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                else:
                    cv2.putText(frame, "Not Smiling", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)

                # increment counter
                face_index += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    test_recognition(0.3, 0.05)
