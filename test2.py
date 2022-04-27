import numpy as np
import cv2 as cv
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as pyplot
import coremltools

output_source = './Hackathon_Sample/Ground_truth'
input_source = './Hackathon_Sample/Images'

data_count = len([name for name in os.listdir(output_source)])


batch_outputs = np.array([cv.imread(
    f"{output_source}/mask_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)]).reshape((data_count, 512, 512, 1))

# 0dim - photo type, 1dim-batch
batch_inputs = np.stack(np.array([[cv.imread(
    f"{input_source}/Aspect/aspect_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
        f"{input_source}/DTM/dtm_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
            f"{input_source}/Flow_Accum/flowacc_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                f"{input_source}/Flow_Direction/flowdir_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                    f"{input_source}/Orthophoto/ortho_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                        f"{input_source}/Prof_curv/pcurv_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                            f"{input_source}/Slope/slope_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                                f"{input_source}/Tang_curv/tcurv_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                                    f"{input_source}/Topo_Wetness/twi_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)]]), axis=-1)

print("end")

# img = batch_inputs[0, :, :, 4]
img = cv.imread('image (3).png', cv.IMREAD_GRAYSCALE)

model=coremltools.models.MLModel("./coreML/archeoHackaton_yolo2_ortho_v5.mlmodel")
model = coremltools.convert(model)
res = model.predict(img)

cv.imshow("test", img)
cv.waitKey()

img = cv.GaussianBlur(img,(9,9),15)

cv.imshow("test", img)
cv.waitKey()

img = cv.Sobel(img,cv.CV_32F,1,1,ksize=5)

cv.imshow("test", img)
cv.waitKey()

img = cv.GaussianBlur(img,(11,11),4)

cv.imshow("test", img)
cv.waitKey()

ret,img = cv.threshold(img,15,200,cv.THRESH_BINARY)

cv.imshow("test", img)
cv.waitKey()

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=6)

cv.imshow("test", img)
cv.waitKey()

# define the true objective function
def objective1(x, a, b):
	return a * x + b

# define the true objective function
def objective2(x, a, b, c):
	return a * x + b * x**2 + c

ax = []
ay = []
for i, width in enumerate(img):
    for j, col in enumerate(width):
        if col != 0:
            ax.append(j)
            ay.append(i)
            break

popt, _ = curve_fit(objective1, ax, ay)

seg = np.zeros((img.shape[0],img.shape[1]))
x_line = np.arange(img.shape[1])
y_line = objective1(x_line, *popt).astype(dtype='int32')
print("etts")
seg[y_line, x_line] = 1
# create a line plot for the mapping function


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
seg = cv.dilate(seg,kernel,iterations = 3)


cv.imshow("test", seg)
cv.waitKey()
