
import cv2
import matplotlib.pyplot as plt

#Step 1

image = cv2.imread("breast_cancer_case.jpg")#read the image

original_image =image

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)


#Step 1 - Show the Original Image, Binary Mask, Infected Area

fig = plt.figure(figsize=(15, 10))

#Adding a subplot at the 1st position 
fig.add_subplot(1, 3, 1)

plt.xlabel('Original Image')
plt.imshow(original_image,cmap="gray")
 
# Adding a subplot at the 2 2nd position
fig.add_subplot(1, 3, 2)

plt.xlabel('Binary Mask')
plt.imshow(binary,cmap="gray")
  
# Adding a subplot at the 3rd position
fig.add_subplot(1, 3, 3)
  
plt.xlabel("Infected Area")
plt.imshow(image,cmap="gray")

plt.show()
