import tkinter as tk
from tkinter import ttk
import os

win=tk.Tk()
win.title("Assignment 2")
win.geometry("500x400")

#T = tk.Text(win, height = 35, width = 70)
 
# Create label
l = tk.Label(win, text = "Please select 8 points for \n transformed image & affine rectification")
l.config(font =("Courier", 14))

def run1():
    os.system("python qn1.py")
def run2():
    os.system("python qn2_corner.py")
def run3():
    os.system("python qn2_edge.py")
def run4():
    os.system("python qn3.py")
def run5():
    os.system("python qn4_tk.py")   

button1=ttk.Button(win,text="Foreground extraction",command=run1)
button2=ttk.Button(win,text="Identify Corner Points",command=run2)
button3=ttk.Button(win,text="Identify Boundary Edges",command=run3)
button4=ttk.Button(win,text="displaying the transformed image",command=run4)
button5=ttk.Button(win,text="Affine rectification",command=run5)

l.pack()
#T.pack()
button1.pack(pady=10)
button2.pack(pady=10)
button3.pack(pady=10)
button4.pack(pady=10)
button5.pack(pady=10)

win.mainloop()