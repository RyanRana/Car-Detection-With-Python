%matplotlib inline
import cv2

from matplotlib import pyplot as plt
img_file  = "car.jpg"
img = cv2.imread(img_file,1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
class_file = 'cars.xml'
car_cascade = cv2.CascadeClassifier(class_file)
cars = car_cascade.detectMultiScale(gray,1.1,1)
# Draw border
ncars=0
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
# Show image
plt.figure(figsize=(10,20))
plt.imshow(img)
