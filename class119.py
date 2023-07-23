import cv2
video = cv2.VideoCapture("bb3.mp4")


tracker = cv2.TrackerCSRT_create()
#print(tracker)

returned,myimage = video.read()
bbox = cv2.selectROI("Tracking", myimage, False)
tracker.init(myimage, bbox)
print(bbox)

#(1003, 416, 128, 137)

def draw_box(myimage, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(myimage, (x,y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(myimage, "tracing", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)


while True:
    dummy, frame = video.read()

    success, detectionbox = tracker.update(frame)
    if success == True:
        draw_box(frame, detectionbox)
    else:
        cv2.putText(myimage, "no tracking", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("bbox", frame)
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()