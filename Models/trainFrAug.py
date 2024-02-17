import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime
import pickle
import time
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import Toplevel, Text, messagebox
import pandas as pd
import os
import pandas as pd
import cv2
from PIL import ImageTk, Image
import csv
from datetime import datetime
import os
import csv
from datetime import datetime
import os
import pickle
from tkinter import messagebox



def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise ValueError("Must supply knn classifier either through knn_clf or model_path")

    # Load a pre-trained KNN classifier if one was passed in
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(X_frame)
    faces_encodings = face_recognition.face_encodings(X_frame, face_locations)

    # Use the KNN model to find the best matches for the test face
    predictions = []
    if len(faces_encodings) > 0:
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)[0]
        are_matches = closest_distances <= distance_threshold
        predictions = [(pred if rec else "Unknown", loc, dist, rec) for pred, loc, dist, rec in
                       zip(knn_clf.predict(faces_encodings), face_locations, closest_distances, are_matches)]
       


    return predictions


def show_prediction_labels_on_image(img, predictions, file_path="C:/Users/shubh/Documents/Visual Studio 2022/FR1/Attendance.csv"):
    img_copy = img.copy()
    for name, (top, right, bottom, left), distance, rec in predictions:
        # Draw a box around the face
        if rec:
            cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2) # Green box for recognized face
        else:
            cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 0, 255), 2) # Red box for unknown face
        
        # Draw a label with a name below the face
        cv2.rectangle(img_copy, (left, bottom - 35), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        if rec:
            cv2.putText(img_copy, f"{name} ({distance[0]:.2f})", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            markAttendance(name, file_path="C:/Users/shubh/Documents/Visual Studio 2022/FR1/Attendance.csv") # call markAttendance only when the face is recognized

        else:
            cv2.putText(img_copy, "Unknown", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return img_copy





import csv
from datetime import datetime
import os

def markAttendance(name, file_path="C:/Users/shubh/Documents/Visual Studio 2022/FR1/Attendance.csv"):
    if not isinstance(name, str):
        raise TypeError("Name must be a string")

    # Create file if it doesn't exist
    file_exists = os.path.isfile(file_path)

    # Get current date and time outside of try block
    now = datetime.now()
    current_time = now.strftime('%I:%M:%p')
    date = now.strftime('%d-%B-%Y')

    try:
        with open(file_path, 'a', newline='') as f:
            # Create a DictWriter object with the fieldnames
            fieldnames = ['Name', 'Time', 'Date']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header row if it doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the new row with the current time and date
            writer.writerow({'Name': name, 'Time': current_time, 'Date': date})
            print(f"{name} was marked as present at {current_time} on {date} in {file_path}")

    except OSError as e:
        raise Exception(f"Could not write to file: {e}")


def start_webcam(model_path="trained_knn_model.clf", fps=30, skip_frames=1):
    # Load face encodings from the saved dataset
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Create a video capture object to read from webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)  # Set frame rate to given value

    process_this_frame = 0
    while True:
        ret, frame = cap.read()
        
        # Check if frame was successfully read from webcam
        if not ret:
            messagebox.showerror("Error", "Failed to read frame from webcam")
            break

        # Only process every skip_frames-th frame of video to save time
        if process_this_frame % skip_frames == 0:
            # Find all the faces and face encodings in the current frame of video
            predictions = predict(frame, knn_clf=knn_clf)

            img = show_prediction_labels_on_image(frame, predictions)
            
            # Check if image was successfully displayed
            if img is not None:
                cv2.imshow('Webcam', img)
            else:
                messagebox.showerror("Error", "Failed to display image")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        process_this_frame = (process_this_frame + 1) % skip_frames

    cap.release()
    cv2.destroyAllWindows()






def check_attendance():
    try:
        # Load attendance file and sort by date and time
        file_path = "C:/Users/shubh/Documents/Visual Studio 2022/FR1/Attendance.csv"

        # Check if file exists and is not empty
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            # Read the existing attendance file and sort it
            attendance = pd.read_csv(file_path)
            attendance = attendance.sort_values(by=['Date', 'Time'], ascending=False)

            # Filter out duplicates based on name and time
            attendance = attendance.drop_duplicates(subset=['Name', 'Time'])

        else:
            raise FileNotFoundError

        # Display attendance data in a new window
        attendance_window = Toplevel()
        attendance_window.title("Attendance Report")
        attendance_text = Text(attendance_window)
        attendance_text.insert("end", attendance.to_string(index=False))
        attendance_text.pack(fill="both", expand="yes")

        # Perform basic regularization and save the updated attendance file
        attendance['Name'] = attendance['Name'].str.title()
        attendance.to_csv(file_path, index=False)

    except FileNotFoundError:
        messagebox.showerror("Error", "Attendance file not found")
    except Exception as e:
        messagebox.showerror("Error", f"Could not load attendance file: {e}")



import random


def capture_images(name):
    # Set up the face detector and video capture device
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    # Create a directory for the user's images
    directory = f"C:/Users/shubh/Documents/Visual Studio 2022/FR1/images/Train2/{name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Capture 15 color images with a 1 second interval
    count = 0
    while count < 15:
        ret, frame = cap.read()
        filepath = os.path.join(directory, f"{count+1}.jpg")
        cv2.imwrite(filepath, frame)
        count += 1
        time.sleep(1)  # Wait for 1 second between captures

    # Release the capture device
    cap.release()

    # Augment the captured images
    for i in range(1, 16):
        # Load the original image
        filepath = os.path.join(directory, f"{i}.jpg")
        img = cv2.imread(filepath)

        for j in range(2):
            # Apply random brightness adjustment
            brightness_offset = random.uniform(-50, 50)
            bright_img = np.clip(img.astype(int) + brightness_offset, 0, 255).astype(np.uint8)

            # Apply random rotation (between -15 and 15 degrees)
            rotation_angle = random.randint(-15, 15)
            center = (int(img.shape[1]//2), int(img.shape[0]//2))
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            rotated_img = cv2.warpAffine(bright_img, M, (img.shape[1], img.shape[0]))

            # Apply random vertical flip
            flip = random.choice([True, False])
            if flip:
                flipped_img = cv2.flip(rotated_img, 1)
            else:
                flipped_img = rotated_img

            # Save the augmented image
            count += 1
            filepath = os.path.join(directory, f"{count}.jpg")
            cv2.imwrite(filepath, flipped_img)

    # Close all windows
    cv2.destroyAllWindows()

def on_register_button_click():
    # Open a new window for registration
    register_window = Toplevel()
    register_window.title("Register New User")

    # Create a label and entry box for the name
    name_label = ttk.Label(register_window, text="Enter your name:", background="white")
    name_label.pack(pady=20)
    name_entry = ttk.Entry(register_window)
    name_entry.pack()

    def on_capture_button_click():
        # Get the user's name and start capturing images
        name = name_entry.get()
        capture_images(name)

    # Create a button to capture images
    capture_button = ttk.Button(register_window, text="Capture Images", command=on_capture_button_click, style='my.TButton')
    capture_button.pack(pady=20)

def start_gui():
    # Create a Tkinter window
    window = Tk()

    # Set the title of the window
    window.title("Attendance System")

    # Set the dimensions of the window
    window.geometry("600x400")
    
    # Load the background image
    bg_image = Image.open('C:/Users/shubh/Documents/Visual Studio 2022/FR1/images/background.jpg')
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Create a canvas and show the background image
    canvas = Canvas(window, width=600, height=400, bg='#2F4F4F', bd=0, highlightthickness=0)
    canvas.create_image(10, 10, image=bg_photo, anchor=NW)
    canvas.pack(padx=10, pady=10)

    # Use a custom font for the labels and buttons
    FONT = ("Helvetica", 14)
    BIG_FONT = ("Helvetica", 18)

    # Create a frame to hold the buttons using grid layout
    button_frame = ttk.Frame(canvas, borderwidth=2, relief='ridge')
    button_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

    # Configure the style of the button frame to have a darkslate gray background
    style = ttk.Style()
    style.configure('TFrame', background='#2F4F4F')

    # Create a label for the frame
    label = ttk.Label(button_frame, text="Choose an option:", font=BIG_FONT)
    label.grid(row=0, column=0, columnspan=4, padx=10, pady=20, sticky="N")

    # Create a button to start the webcam
    start_button = ttk.Button(button_frame, text="Start Webcam", command=start_webcam, style='my.TButton')
    start_button.grid(row=1, column=1, padx=20, pady=10)

    # Create a button to check attendance
    attendance_button = ttk.Button(button_frame, text="Check Attendance", command=check_attendance, style='my.TButton')
    attendance_button.grid(row=2, column=1, padx=20, pady=10)

    # Create a button to register a new user
    register_button = ttk.Button(button_frame, text="Register New User", command=on_register_button_click, style='my.TButton')
    register_button.grid(row=3, column=1, padx=20, pady=10)

    # Create a button to quit the application
    quit_button = ttk.Button(button_frame, text="Quit", command=window.quit, style='my.TButton')
    quit_button.grid(row=4, column=1, padx=20, pady=10)

    # Use a custom style for the buttons
    style.configure('my.TButton', font=FONT, foreground="black", background='#0C2340', padding=10, width=15)

    style.map('my.TButton', background=[('active', '#FFFFFF')])

    # Run the Tkinter event loop
    window.mainloop()

start_gui()

