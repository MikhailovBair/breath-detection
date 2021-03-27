import cv2
import json
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from pulse_detector import pulse_from_sample, crop_fragment

DETECTION_CHANGE_PERIOD = 120
FPS = 30
name = sys.argv[1]
extension = sys.argv[2]
should_not_process_video = int(sys.argv[3])
video_name = "videos/" + name + "." + extension
breath_changes_name = "breath_changes_" + name + ".json"
cap = cv2.VideoCapture(video_name)

# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

chest_sizes = np.array([0])
breath_changes = np.array([0])


def process_video():
    x, y, w, h = 0, 0, 0, 0
    i = 0
    last_time = -DETECTION_CHANGE_PERIOD
    frames = []
    frames_max = 150

    pulse = 0
    breath = 0

    while True:
        global chest_sizes, breath_changes
        i += 1
        ret, frame = cap.read()
        if not ret:
            with open(breath_changes_name, "w") as write_file:
                json.dump(breath_changes.tolist(), write_file)
            break

        if i - last_time >= DETECTION_CHANGE_PERIOD:
            faces = face_cascade.detectMultiScale(frame, 1.1, 10, minSize=(100, 100))
            if faces is None or len(faces) == 0:
                print("Failed to detect face")
                continue
            x, y, w, h = faces[0]
            last_time = i

        face = crop_fragment(frame, faces[0])
        face = np.mean(face[:, :, 1])
        if len(frames) < frames_max:
            frames = frames[0:] + [face]
        else:
            frames = frames[1:] + [face]

        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        crop_img = frame[
            y + round(1.2 * h) : y + round(3 * h),
            x - round(w * 0.3) : x + round(w * 1.4),
        ]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        frame = cv2.rectangle(
            frame,
            (x - round(w * 0.3), y + round(1.2 * h)),
            (x + round(w * 1.4), y + round(3 * h)),
            (255, 255, 255),
            2,
        )
        frame = cv2.rectangle(
            frame,
            (x + w // 3, y + h // 15),
            (x + 2 * w // 3, y + h // 5),
            (255, 255, 255),
            2,
        )

        mean = np.mean(crop_img, (0, 1), keepdims=True)
        cnt = np.sum((np.sum(np.abs(crop_img - mean), axis=2) < 40))

        chest_sizes = np.append(
            chest_sizes, cnt / (np.shape(crop_img)[0] * np.shape(crop_img)[1])
        )
        if abs(chest_sizes[-1] - chest_sizes[-2]) < 0.01:
            breath_changes = np.append(
                breath_changes, breath_changes[-1] + chest_sizes[-1] - chest_sizes[-2]
            )

        if i % 10 == 0:
            sz = len(breath_changes)
            plt.plot(np.arange(max(sz - 50, 0), sz), breath_changes[-50:])
            plt.xlim(max(sz - 50, 0), sz)
            plt.pause(0.001)

            if len(frames) == frames_max:
                pulse = pulse_from_sample(frames) * 60
                breath = get_frequency()

        cv2.rectangle(frame, (0, 0), (400, 100), (255, 255, 255), cv2.FILLED)
        cv2.putText(
            frame,
            "pulse: {:.2f}".format(pulse),
            (10, 30),
            cv2.FONT_ITALIC,
            1.1,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "breathing rate: {:.2f}".format(breath),
            (10, 60),
            cv2.FONT_ITALIC,
            1.1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def get_frequency(graph=False):
    global breath_changes
    if should_not_process_video:
        with open(breath_changes_name, "r") as read_file:
            breath_changes = np.array(json.load(read_file))

    # plt.plot(np.arange(len(breath_changes)), breath_changes)
    #
    one_chunk = np.tile(breath_changes, 10)
    y = fft(one_chunk - one_chunk.mean())
    x = fftfreq(len(one_chunk), 1 / FPS)
    mask = x >= 0.28
    frequency = x[mask][abs(y[mask]).argmax()]
    return frequency


def make_verdict(freq, times):
    breath_in_minute = freq * 60
    if 15 <= breath_in_minute <= 20:
        verdict = "Your breath is normal"
    elif breath_in_minute > 20:
        verdict = "You breath heavy"
    else:
        verdict = "You breath lightly"

    while True:
        ans = np.zeros((130, 900, 3))

        cv2.putText(
            ans,
            "Your breath rate is: {:.2} breathes per sec".format(freq),
            (10, 30),
            cv2.FONT_ITALIC,
            1.1,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            ans,
            "You have breathed {:.0f} times".format(times / FPS * freq),
            (10, 60),
            cv2.FONT_ITALIC,
            1.1,
            (0, 255, 0),
            2,
        )
        cv2.putText(ans, verdict, (10, 90), cv2.FONT_ITALIC, 1.1, (0, 0, 255), 2)
        cv2.imshow("Frame", ans)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


if not should_not_process_video:
    process_video()

cap.release()
cv2.destroyAllWindows()

freq = get_frequency(True)
make_verdict(freq, len(breath_changes))
