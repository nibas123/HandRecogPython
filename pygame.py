import cv2
import mediapipe as mp
import time
import keyboard

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

hand_detected = False

while True:
    success, img = cap.read()
    
    if not success:
        print("Error: Failed to read frame from the webcam.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hand_detected = False  # Reset flag for current frame

    if results.multi_hand_landmarks:
        hand_detected = True  # Set flag if hand gestures are detected
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get the coordinates of specific landmarks
            index_tip = (handLms.landmark[8].x * w, handLms.landmark[8].y * h)
            thumb_tip = (handLms.landmark[4].x * w, handLms.landmark[4].y * h)

            # Calculate the distance between index and thumb tips
            distance = ((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5

            # Perform keyboard actions based on hand gestures
            if distance < 50:  # Adjust threshold as needed
                keyboard.release('left')
                keyboard.press('right')
            else:
                keyboard.release('right')
                keyboard.press('left')

    # Release any pressed keys if no hand gestures are detected
    if not hand_detected:
        keyboard.release('right')
        keyboard.release('left')

    # Display FPS and image
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
