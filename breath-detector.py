import cv2
import json
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

name = sys.argv[1]
extension = sys.argv[2]
should_not_process_video = int(sys.argv[3])
print(name)
video_name = "videos/" + name + "." + extension
times_name = "times_" + name + ".json"
breath_changes_name = "breath_changes_" + name + ".json"
cap = cv2.VideoCapture(video_name)

# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

fig = plt.figure()
ax = fig.add_subplot(111)

chest_sizes = np.array([0])
times = np.array([0])
breath_changes = np.array([0])

i = 0
x, y, w, h = 0, 0, 0, 0
last_time = -10000


def process_video():
    while True:
        global i, last_time, chest_sizes, times, breath_changes
        i += 1
        ret, frame = cap.read()
        if not ret:
            break

        if i - last_time >= 120:
            faces = face_cascade.detectMultiScale(frame, 1.1, 10, minSize=(100, 100))
            if faces is None or len(faces) == 0:
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
        if abs(chest_sizes[-1] - chest_sizes[-2]) < 0.01:
            breath_changes = np.append(breath_changes, breath_changes[-1] + chest_sizes[-1] - chest_sizes[-2])
            times = np.append(times, i / 30)

        ax.plot(times, breath_changes)
        fig.canvas.draw()
        plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.cla()
        cv2.imshow('Graph', plot_img_np)
        cv2.imshow('Crop', crop_img)

        print(i / 30, i)

        if i % 100 == 0:
            with open(breath_changes_name, "w") as write_file:
                json.dump(breath_changes.tolist(), write_file)

            with open(times_name, "w") as write_file:
                json.dump(times.tolist(), write_file)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_frequency():
    global times, breath_changes
    if should_not_process_video:
        with open(times_name, "r") as read_file:
            times = np.array(json.load(read_file)) * 30
        times = np.round(times).astype('int64')

        with open(breath_changes_name, "r") as read_file:
            breath_changes = np.array(json.load(read_file))

    plt.plot(times, breath_changes)
    plt.figure(figsize=(12, 12))
    one_chunk = np.tile(breath_changes, 10)
    y = fft(one_chunk - one_chunk.mean())
    x = fftfreq(len(one_chunk), 1 / 30)
    plt.plot(x, np.abs(y))
    plt.xlim(.1, 3)
    plt.ylim(0, 100)
    plt.show()
    mask = (x >= .27)
    frequency = x[mask][abs(y[mask]).argmax()]
    return frequency


if not should_not_process_video:
    process_video()

cap.release()
cv2.destroyAllWindows()


freq = get_frequency()
print(freq, len(times) / 30 * freq)
