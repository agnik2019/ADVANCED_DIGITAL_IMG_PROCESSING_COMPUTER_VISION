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

source_vertices = []
destination_vertices = []
image_patch = None
click_count = 0

# Image read for displaying in GUI Window
image = Image.open(filename)
window_image = ImageTk.PhotoImage(image)

# Creating the canvas object
canvas = Canvas(tinker_root, bg="black", width=window_image.width(), height=window_image.height())
canvas.pack(expand=YES, fill=BOTH)
canvas_image = canvas.create_image(0, 0, image=window_image, anchor=NW)


def handle_source_point(coordinates):
    global source_vertices

    canvas.create_oval(coordinates.x - 3, coordinates.y - 3, coordinates.x + 3, coordinates.y + 3, fill="red")
    source_vertices.append(np.array([coordinates.x, coordinates.y]))


def handle_destination_point(coordinates):
    global destination_vertices

    canvas.create_oval(coordinates.x - 3, coordinates.y - 3, coordinates.x + 3, coordinates.y + 3, fill="blue")
    destination_vertices.append(np.array([coordinates.x, coordinates.y]))


def get_image_patch():
    global image_patch
    global img

    dummy_image = np.zeros(img.shape, dtype=img.dtype)
    cv2.fillPoly(dummy_image, np.array([source_vertices]), (1, 1, 1))
    image_patch = img * dummy_image
    # print image_patch


def transform_and_paste_patch():
    global img
    global image
    global image_patch
    global window_image
    global canvas_image

    dummy_image = np.ones(img.shape, dtype=img.dtype)
    cv2.fillPoly(dummy_image, np.array([destination_vertices]), (0, 0, 0))
    img = img * dummy_image
    # Using in built function from opencv to find homography
    homography = cv2.findHomography(np.array(source_vertices), np.array(destination_vertices))[0]
    print(homography)
    transformed_image = cv2.warpPerspective(image_patch, homography, (img.shape[1], img.shape[0]))
    img = img + transformed_image

    image = Image.frombytes('RGB', (img.shape[1], img.shape[0]), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image.save("output.jpg")
    window_image = ImageTk.PhotoImage(image)
    canvas_image = canvas.itemconfig(canvas_image, image=window_image)
    canvas.pack()


def save_output():
    image.save("output.jpg")


def on_click(coordinates):
    global click_count

    click_count += 1
    if click_count <= 4:
        # vertex selected is of source
        handle_source_point(coordinates)
    elif click_count <= 8:
        # vertex selected is of destination
        handle_destination_point(coordinates)
    else:
        # Image patch already transformed
        return

    if click_count == 4:
        # Store the patch of image to be transformed
        get_image_patch()
    if click_count == 8:
        transform_and_paste_patch()
        save_output()


# Adding on_click event to left mouse button click
canvas.bind("<Button-1>", on_click)
tinker_root.mainloop()
