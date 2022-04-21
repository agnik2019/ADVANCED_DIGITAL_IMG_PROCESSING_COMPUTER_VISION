import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

tinker_root = Tk()
tinker_root.resizable(0, 0)
tinker_root.title("Main Window")

# Change input file name here
filename = 'PataChitraPuri_1.jpg'
img = cv2.imread(filename, cv2.IMREAD_COLOR)

list1 = []
list2 = []
image_patch = None
click_count = 0

# Image read for displaying in GUI Window
image = Image.open(filename)
window_image = ImageTk.PhotoImage(image)

# Creating the canvas object
canvas = Canvas(tinker_root, bg="black", width=window_image.width(), height=window_image.height())
canvas.pack(expand=YES, fill=BOTH)
canvas_image = canvas.create_image(0, 0, image=window_image, anchor=NW)


# affine rectification
# This function returns the vanishing line equation
def vanishingline(list1, list2):
    
    # Equation of lines
    l1 = np.cross([list1[0][0], list1[0][1], 1], [list1[1][0], list1[1][1], 1])
    l2 = np.cross([list1[2][0], list1[2][1], 1], [list1[3][0], list1[3][1], 1])
    p1 = np.cross([list2[0][0], list2[0][1], 1], [list2[1][0], list2[1][1], 1])
    p2 = np.cross([list2[2][0], list2[2][1], 1], [list2[3][0], list2[3][1], 1])
    
    temp1 = np.cross(l1,l2)
    temp2 = np.cross(p1,p2)
    
    # Vanishing Points
    v1 = temp1/temp1[2]
    v2 = temp2/temp2[2]
    print(v1, v2)
    
    return np.cross(v1, v2)
    


# Returns the affine rectification homography matrix
def affineHom(line):    
    return np.array([[1, 0, 0], [0, 1, 0], [line[0]/line[2], line[1]/line[2], 1]])



def parallel_line_1(coordinates):
    global source_vertices

    canvas.create_oval(coordinates.x - 3, coordinates.y - 3, coordinates.x + 3, coordinates.y + 3, fill="red")
    list1.append([coordinates.x, coordinates.y])


def parallel_line_2(coordinates):
    global destination_vertices

    canvas.create_oval(coordinates.x - 3, coordinates.y - 3, coordinates.x + 3, coordinates.y + 3, fill="blue")
    list2.append([coordinates.x, coordinates.y])

def main_helper():
    global img
    global image
    global image_patch
    global window_image
    global canvas_image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size = gray.shape
    sizeNew = (size[1], size[0])
    vLine = vanishingline(list1, list2)
    HomMatrix = np.float32(affineHom(vLine))
    img_affine = cv2.warpPerspective(img,HomMatrix, sizeNew)
    image = Image.frombytes('RGB', (img_affine.shape[1], img_affine.shape[0]), cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB))
    image.save("AffineRectifiedImage.jpg")
    window_image = ImageTk.PhotoImage(image)
    canvas_image = canvas.itemconfig(canvas_image, image=window_image)
    canvas.pack()

def save_output():
    image.save("AffineRectifiedImage.jpg")


def on_click(coordinates):
    global click_count

    click_count += 1
    if click_count <= 4:
        # vertex selected is of source
        parallel_line_1(coordinates)
    elif click_count <= 8:
        # vertex selected is of destination
        parallel_line_2(coordinates)
    else:
        # Image patch already transformed
        return

    if click_count == 8:
        main_helper()
        save_output()


# Adding on_click event to left mouse button click
canvas.bind("<Button-1>", on_click)
tinker_root.mainloop()
