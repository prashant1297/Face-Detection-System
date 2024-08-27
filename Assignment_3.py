'''
Face detection from a live web camera feed
'''

import cv2 as cv

# Loading Frontal face detecting classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

fourcc = cv.VideoWriter.fourcc(*'MJPG') # Video codec specifier

# Video writer object to save web cam video
output_vdo_filename = 'WebCam_vdo.avi'
output = cv.VideoWriter(output_vdo_filename, fourcc, 15, (640, 480))

vdo = cv.VideoCapture(0) # Accessing web cam
frame_count = 0

print(f'Press "s" to exit the live cam')

# Reading and Processing web cam frames
while True:
    flag, frame = vdo.read()
    frame_count += 1
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7) # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    for (x,y,w,h) in faces:                                                            # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
    if not flag:
        break
    cv.putText(frame, f'Frame no.= {frame_count}', (0, 30), 1, 1, (0, 255, 0), 1)
    cv.imshow(f'Live Web Cam', frame)
    output.write(frame)
    if cv.waitKey(1) & 0xFF == ord('s'):
        break

vdo.release()
cv.destroyAllWindows()
print(f'Web cam video saved as {output_vdo_filename}')