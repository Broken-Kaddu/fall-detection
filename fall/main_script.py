import cv2
import time

# Function to detect motion in the video frames
def detect_motion(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust this threshold as needed
            motion_detected = True
            break

    return motion_detected

# Function to detect falls
def detect_fall():
    cap = cv2.VideoCapture(0)

    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Couldn't read video source")
        return

    no_motion_time = 0
    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Check for motion
        motion_detected = detect_motion(frame1, frame2)

        if motion_detected:
            print("Motion detected!")
            no_motion_time = 0
        else:
            no_motion_time += 1
            print("No motion detected for {} seconds".format(no_motion_time))

        if no_motion_time >= 5:
            print("Fall detected!")
            no_motion_time = 0

        # Update frame1 to next frame
        frame1 = frame2

        # Display the frames
        cv2.imshow('Video', frame2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Call the function to detect falls
    detect_fall()