import mediapipe as mp 
import numpy as np 
import cv2 

# Initialize the webcam capture (index 0)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Input to name the saved data file
name = input("Enter the name of the data: ")

# Initialize Mediapipe holistic model and drawing utilities
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Data storage for landmark points
X = []
data_size = 0

# Main loop to capture and process frames
while True:
    lst = []

    # Capture a frame from the webcam
    ret, frm = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame for a mirror effect
    frm = cv2.flip(frm, 1)

    # Process the frame using Mediapipe holistic model
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if face landmarks are detected
    if res.face_landmarks:
        # Add face landmarks relative to nose (landmark 1)
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Add left hand landmarks if detected
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Append zeros if left hand not detected

        # Add right hand landmarks if detected
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Append zeros if right hand not detected

        # Add the landmarks to the data list
        X.append(lst)
        data_size += 1

    # Draw the detected landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the data size on the frame
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow("window", frm)

    # Exit if 'ESC' is pressed or data size exceeds 199
    if cv2.waitKey(1) == 27 or data_size > 120:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save the captured data to a NumPy file
np.save(f"{name}.npy", np.array(X))
print(f"Data saved to {name}.npy with shape {np.array(X).shape}")
