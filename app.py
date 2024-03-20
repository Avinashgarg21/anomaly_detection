import cv2
import time
import winsound  # for alarm purpose, if any movement is detected across web cam in this case the alarm will be fired

# for initializing background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize motion event log file, this log file will have all the timestamp at which the movement is detected in the cam, along with the anomaly percentage.
log_file = open('motion_events.log', 'a')

# apth for the log file
audio_file_path = 'notification.wav'

# in order to play the sound continuously till the movement is detected
winsound.PlaySound(audio_file_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

while True:
    # for reading a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Applying background subtraction to detect motion
    fgMask = backSub.apply(frame)

    # calculating confidence score , confidence score is calculated based upon what proportion of pixel is changing its state from previous state
    num_pixels = cv2.countNonZero(fgMask)
    total_pixels = frame.shape[0] * frame.shape[1]
    confidence_score = num_pixels / total_pixels

    # for display confidence score on the cam/frame
    cv2.putText(frame, f'Confidence: {confidence_score:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # we put the confidence score as 0.1, we can play around this threshold value, but 0.1 worked fine for me.
    # we can adjust this threshold if needed
    if confidence_score > 0.1:
        # for saving timestamp when the motion detected
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'Motion detected at {timestamp} with confidence score {confidence_score}\n')

    # show the frame with detected result, that what is the confidence
    cv2.imshow('Motion Detection', frame)

    # break the loop if 'z' is pressed
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

# stop playing the sound
winsound.PlaySound(None, winsound.SND_PURGE)
cap.release()
log_file.close()
cv2.destroyAllWindows()