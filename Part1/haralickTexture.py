from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True)
ap.add_argument("-t", "--test", required=True)

args = vars(ap.parse_args())

print("[INFO] extracting features...")
data = []
labels = []

for imagePath in glob.glob(args["training"] + "/*.jpg"):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = imagePath[imagePath.rfind("/") + 1:].split("-")[0]

    features = mahotas.features.haralick(image).mean(axis=0)

    data.append(features)
    labels.append(texture)

print("[INFO] traning model...")
model = LinearSVC()
model.fit(data, labels)
print("[INFO] classifying...")

for imagePath in glob.glob(args["test"] + "/*.jpg"):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = mahotas.features.haralick(gray).mean(axis=0)

    pred = model.predict(features.reshape(1, -1))[0]
    cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Image", image)
    cv2.waitKey(0)