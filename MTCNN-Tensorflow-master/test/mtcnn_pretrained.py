
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
 
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
 # load the image
 data = pyplot.imread(filename)
 # plot the image
 pyplot.imshow(data)
 # get the context for drawing boxes
 ax = pyplot.gca()
 # plot each box
 for result in result_list:
 # get coordinates
    x, y, width, height = result['box']
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)
 # draw the dots
    for key, value in result['keypoints'].items():
    # create and draw dot
        dot = Circle(value, radius=2, color='red')
        ax.add_patch(dot)
 # show the plot
 pyplot.show()

def draw(image, result_list):
 # load the image
 data = image
 # plot the image
 pyplot.imshow(data)
 # get the context for drawing boxes
 ax = pyplot.gca()
 # plot each box
 for result in result_list:
 # get coordinates
    x, y, width, height = result['box']
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)
 # draw the dots
    for key, value in result['keypoints'].items():
    # create and draw dot
        dot = Circle(value, radius=2, color='red')
        ax.add_patch(dot)
 # show the plot
 pyplot.show()
 
filename = "../../DATA/test/test1.jpg"
# load image from file
pixels = pyplot.imread(filename)
"""
path = "../../DATA/test/test1.jpg"
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
"""
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:
    print(face)
# display faces on the original image
#draw_image_with_boxes(filename, faces)
draw(pixels, faces)