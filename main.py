import cv2                                  # for detection purposes
import numpy as np                          # for flattening or conversion of matrix 
import tkinter as tk
from threading import Thread, Event
import pygame
import time

# Sound functions
pygame.mixer.init()
beep_sound = pygame.mixer.Sound(r"beep.mp3")
alarm_sound = pygame.mixer.Sound(r"Alarm.mp3")

#Initialization of DNN 
net = cv2.dnn.readNet(
    r"yolov3.weights",
    r"yolov3.cfg",                                      # outsourced from GitHUb
)

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
classes = []
with open(r"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Setting the object which needs to be monitored
object_index = classes.index("remote")

stop_event = Event()

# Function to play beep sound 
def play_beep(volume):
    beep_sound.set_volume(volume)
    beep_sound.play()

# Function to play alarm sound
def play_alarm(volume):
    alarm_sound.set_volume(volume)
    alarm_sound.play()

# Function for object detection
def detect_objects(volume_control, stop_event):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    initial_position = None
    object_detected_last_frame = False

    while not stop_event.is_set():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Resizing the frame for faster processing
        input_size = 416  # YOLO input size
        small_frame = cv2.resize(frame, (input_size, input_size))

        height, width, channels = small_frame.shape

        blob = cv2.dnn.blobFromImage(
        small_frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        
        try:
            outs = net.forward(output_layers)
        except cv2.error as e:
            print(f"Error during forward pass: {e}")
            break

        # To show the Information on the screen
        class_ids = []
        confidences = []
        boxes = []

        for frame_out in outs:
            for detection in frame_out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == object_index:
                    
                    # Coordinates mapping
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(
            boxes, confidences, score_threshold=0.5, nms_threshold=0.4
        )

        font = cv2.FONT_HERSHEY_PLAIN
        detected = False
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (255, 0, 0)
                # Resize the bounding box back to the original frame size
                x = int(x * (frame.shape[1] / width))
                y = int(y * (frame.shape[0] / height))
                w = int(w * (frame.shape[1] / width))
                h = int(h * (frame.shape[0] / height))

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

                # Check position and adjust volume
                current_position = (center_x, center_y)
                detected = True
                if initial_position is None:
                    initial_position = current_position
                else:
                    distance = np.linalg.norm(
                        np.array(initial_position) - np.array(current_position)
                    )
                    volume = min(
                        1.0, distance / 500
                    )  # Adjust distance scaling as needed
                    play_beep(volume * volume_control.get())

        if detected:
            if not object_detected_last_frame:
                play_beep(volume_control.get())
                pygame.mixer.Sound.stop(alarm_sound)  # Stop the alarm if object is detected again
            object_detected_last_frame = True
        else:
            if object_detected_last_frame:
                play_alarm(volume_control.get())  # Play the alarm if object is lost
            object_detected_last_frame = False

        # Display the detection
        cv2.imshow("Live Camera Feed", frame)

        # Limit the frame rate
        curr_time = time.time() - start_time
        if curr_time < 1.0 / 30:
            time.sleep(1.0 / 30 - curr_time)

        # Thread stops if set 
        if stop_event.is_set():
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Terminate the frames and camera 
    cap.release()
    cv2.destroyAllWindows()

# Function to start the object detection thread
def start_detection():
    global detection_thread, stop_event
    stop_event.clear()
    detection_thread = Thread(target=detect_objects, args=(volume_control, stop_event))
    detection_thread.start()

# Function to stop the detection thread
def stop_detection():
    global stop_event
    stop_event.set()
    detection_thread.join()

# GUI Interface 
root = tk.Tk()
root.title("Object Detection with Buzzer")
root.geometry("1280x720")
root.configure(bg="grey")

label = tk.Label(
    root,
    text="Press the button to start object detection:",
    font=("Consolas", "15"),
    height=3,
    width=1000,
    bg="black",
    fg="white",
)
label.pack()

start_button = tk.Button(
    root,
    text="Start Detection",
    font=("Consolas", "15"),
    command=start_detection,
    bg="black",
    fg="white",
    height=2,
    width=30,
)
start_button.place(x=0, y=656)

exit_button = tk.Button(
    root,
    text="Exit",
    command=stop_detection,
    font=("Consolas", "15"),
    bg="black",
    fg="white",
    height=2,
    width=30,
)
exit_button.place(x=400, y=656)

volume_control = tk.DoubleVar(value=0.5)
volume_slider = tk.Scale(
    root,
    from_=0,
    to=1,
    length=500,
    bg="black",
    fg="white",
    orient=tk.HORIZONTAL,
    resolution=0.01,
    label="Buzzer Volume Control",
    variable=volume_control,
)
volume_slider.place(x=772, y=656)

root.mainloop()