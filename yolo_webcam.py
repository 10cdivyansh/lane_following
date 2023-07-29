from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
THRESHOLD_DISTANCE=500

dist = 0
focal = 450
pixels = 30
width = 4


#cap=cv2.VideoCapture(r"C:\Users\Divyansh\Downloads\pexels-donatello-trisolino-1333384-1920x1080-30fps.mp4")
#cap = cv2.VideoCapture(r"C:\Users\Divyansh\Downloads\highway-10364-720p_ayehbkkP.mp4")
#cap= cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\VID_20230606_203414.mp4")

model =YOLO("../yolo_weights/yolov8n.pt")
classnames=["person","bicycle","car","motorbike","aeroplane","bus","train","truck"
            "boat","traffic light","fire hydrant", "stop sign","parking meter",
            "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
            "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat",
            "baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
            "orange","broccoli","carrot","hot dog","pizza","donut","cake",
            "chair","sofa","potted plant","bed","dining table","toilet","monitor","laptop",
            "mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","fridge","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"]



def calculate_distance(x1,y1,x2,y2):
    # Calculate the Euclidean distance between two points
    distance = width*focal/y1
    return distance
def calculate_angle(x1, y1, x2, y2):
    # Calculate the angle between two points
    midx=(x1+x2)/2
    midy=(y1+y2)/2
    angle = math.degrees(math.atan(midx/midy))
    return angle

def detect_lanes(image, edges):
    # Define parameters for Hough Transform
    rho = 1
    theta = np.pi/180
    threshold = 50  # Adjust the threshold to be more sensitive
    min_line_length = 100 # Adjust the minimum line length
    max_line_gap = 50 # Adjust the maximum line gap

    # Perform probabilistic Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Create an empty array for collecting lane lines
    lane_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_lines.append(line)

    # Create a blank image for drawing lane lines
    line_image = np.zeros_like(image)
    for line in lane_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Combine the line image with the original image
    result = cv2.addWeighted(image, 0, line_image, 1, 0)

    return result


# Read the video file


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred


def detect_edges(image):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    return edges


def region_of_interest(image, vertices):
    # Create a mask for the region of interest
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image



while True:
    success,img=cap.read()
    results=model(img,stream=True)
    is_close = False
    close_direction = ""
    for  r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]

            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
           # print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #bbox = int(x1), int(y1), int(w), int(h)
            distance = calculate_distance(x1, y1, x2, y2)
            angle = calculate_angle(x1, y1, x2, y2)
            w=x2-x1
            h=y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            cvzone.putTextRect(img, f"Distance: {distance:.2f} units", (x1, y1 - 40), scale=1, thickness=1)
            cvzone.putTextRect(img, f"Angle: {angle:.2f} degrees", (x1, y1 - 10), scale=1, thickness=1)
            #cvzone.putTextRect(img, f"Angl: {angle:.2f} degrees", (x1, y1 - 10), scale=1, thickness=1)


            conf=math.floor((box.conf[0]*100))/100
            print(conf)

            cls=int(box.cls[0])
            obj_img=cvzone.putTextRect(img,f"{classnames[cls]},{conf}",(max(0,x1),max(29,y1+10)),scale=1,thickness=1)

            # Check if object is too close
            if distance < THRESHOLD_DISTANCE:
                is_close = True
                if x1 < img.shape[1] / 2:
                    close_direction = "left"
                else:
                    close_direction = "right"

            # Display "stop" if no possible path found
        if is_close:
            cv2.putText(img, f"Close object detected. Turn {close_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        else:
            cv2.putText(img, "No obstacles. Go ahead.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Preprocess the frame
    preprocessed = preprocess_image(img)

    # Detect edges in the preprocessed frame
    edges = detect_edges(preprocessed)

    # Define the region of interest vertices
    height, width = img.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)

    # Apply region of interest masking
    masked_image = region_of_interest(edges, vertices)

    # Detect lanes in the masked image
    lane_img= detect_lanes(img, masked_image)

    # Display the resulting frame
    #cv2.imshow("Image",img)
  #  cv2.imshow("Lane Detection", result)
    #cv2.waitKey(1)
    #cap.release()
    #cv2.destroyAllWindows()
    combined_img = cv2.addWeighted(img, 1, lane_img, 0.5, 0)
    #for r in obj_results:
    #   boxes = r.boxes
    #    for box in boxes:
    #       x1, y1, x2, y2 = box.xyxy[0]
    #      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
    #        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw class label
    #       cls = int(box.cls[0])
    #        conf = math.floor((box.conf[0] * 100)) / 100
    #        cvzone.putTextRect(combined_img, f"{classnames[cls]}, {conf}", (max(0, x1), max(29, y1 - 20)), scale=2,
    #                          thickness=1)

        # Display the combined image
    cv2.imshow("Combined Image", combined_img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

