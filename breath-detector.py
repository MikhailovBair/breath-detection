import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

#cap = cv2.VideoCapture("videos/test.MOV")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

fig = plt.figure()
ax = fig.add_subplot(111)

chest_sizes = np.array([])
times = np.array([])

i = 0
x, y, w, h = 0, 0, 0, 0
last_time = -10000
while (True):
    # Capture frame-by-frame
    i += 1
    ret, frame = cap.read()
    if i - last_time >= 120:
        faces = face_cascade.detectMultiScale(frame, 1.1, 10, minSize=(100, 100))
        if (faces is None or len(faces) == 0):
            print('Failed to detect face')
            continue
        x, y, w, h = faces[0]
        last_time = i


    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_img = frame[y + round(1.2 * h): y + round(3 * h), x - round(w * 0.3): x + round(w * 1.4)]

    sz = np.shape(crop_img)
    mean = np.mean(crop_img, (0, 1), keepdims=True)
    cnt = np.sum((np.sum(np.abs(crop_img - mean), axis=2) < 40))

    chest_sizes = np.append(chest_sizes, cnt / (np.shape(crop_img)[0] * np.shape(crop_img)[1]))
    times = np.append(times, i / 30)

    ax.plot(times, chest_sizes)
    fig.canvas.draw()
    plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()
    cv2.imshow('Graph', plot_img_np)

    #cv2.rectangle(crop_img, (0, 0), (200, 200), mean, cv2.FILLED)
    cv2.imshow('Crop', crop_img)

    print(i / 30, i)

    if i % 100 == 0:
        with open("chest_sizes_120_test.json", "w") as write_file:
            json.dump(chest_sizes.tolist(), write_file)

        with open("times_fixed_120_test.json", "w") as write_file:
            json.dump(times.tolist(), write_file)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
