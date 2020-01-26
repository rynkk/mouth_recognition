from tkinter import *

import PIL
from PIL import Image, ImageTk

import cv2
import dlib
import math
from imutils import face_utils
import numpy as np
import datetime
import random
import tkinter

def euclidian_distance(p1, p2):
    diff_x = abs(p2[0]-p1[0])
    diff_y = abs(p2[1]-p1[1])
    return math.sqrt(diff_x*diff_x + diff_y*diff_y)

class GameBoard(Frame):

    gameboxs = ""

    def __init__(self, parent, rows=7, columns=7, size=49, color1="lightgray", color2="lightgray"):

        '''size is the size of a square, in pixels'''
        self.parent = parent
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        canvas_width = columns * size
        canvas_height = rows * size

        self.delay = 40 ## delay of fps
        #self.vid = MyVideoCapture('vtest.avi')
        self.vid = MyVideoCapture(0)


        self.gameboxs = [[GameBox(i, j, 50, "white") for j in range(columns)] for i in range(rows)]

        color = self.color2

        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2

            for col in range(self.columns):
                self.gameboxs[row][col].setColor(color)
                color = self.color1 if color == self.color2 else self.color2



        for j in range(columns):
            self.gameboxs[0][j] = GameBox(0,j,20,"green", label=True)
            #image = PhotoImage(file="bmp/test.png")
            image = PhotoImage(file="bmp/label_top_"+str(j)+".png")
            self.gameboxs[0][j].setFigureImage(image)

        for i in range(columns):
            self.gameboxs[i][0] = GameBox(i,0,20,"green", label=True)
            image = PhotoImage(file="bmp/label_left_"+str(i)+".png")
            self.gameboxs[i][0].setFigureImage(image)


        Frame.__init__(self, parent)

        ##################################################################################################
        ## Add right side with cam
        self.rightCam = Canvas(self, borderwidth=0, highlightthickness=0,
                             width=self.vid.width+50, height=self.vid.height+50)
        self.rightCam.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self.rightCam.create_rectangle(0, 0, self.vid.width, self.vid.height, outline="black", fill="white",
                                     tags="square")

        ##################################################################################################
        ## Add top and left border

        self.borderTop = Canvas(self, borderwidth=0, highlightthickness=0,
                             width=canvas_width, height=30)
        self.borderTop.pack(side="top", fill="both", expand=True, padx=0, pady=0)

        self.borderTop.create_rectangle(30+size, 0, canvas_width+20, 30, outline="black", fill="white",
                                     tags="square")

        borderTopChars = ["A", "B", "C", "D", "E", "F"]
        i = 0
        for s in borderTopChars:
            self.borderTop.create_text(2*size+i*size,15, font="Times 20 italic bold", text=s)
            i += 1

        self.borderLeft = Canvas(self, borderwidth=0, highlightthickness=0,
                             width=30, height=canvas_height)
        self.borderLeft.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        self.borderLeft.create_rectangle(0, size, 30, canvas_height-10, outline="black", fill="white",
                                     tags="square")


        borderLeftChars = ["1", "2", "3", "4", "5", "6"]
        i = 0
        for s in borderLeftChars:
            self.borderLeft.create_text(15,18+size+i*size, font="Times 20 italic bold", text=s)
            i += 1

        ###################################################################################################

        self.canvas = Canvas(self, borderwidth=0, highlightthickness=0,

                             width=canvas_width, height=canvas_height)

        self.canvas.pack(side="top", fill="both", expand=True, padx=0, pady=0)


        # this binding will cause a refresh if the user interactively
        # changes the window size

        self.canvas.bind("<Configure>", self.refresh)
        self.button = Button(self, text="Click!", command=self.newColor)
        self.button.pack(side=LEFT)

        self.update()



    def newColor(self):
        color = self.color2

        for row in range(self.rows):
            if row == 0:
                continue
            color = self.color1 if color == self.color2 else self.color2

            for col in range(self.columns):
                if col == 0:
                    continue
                self.gameboxs[row][col].setColor(color)
                color = self.color1 if color == self.color2 else self.color2

        randRow = random.randrange(1, 10)
        randColumn = random.randrange(1, 10)
        g = self.gameboxs[randRow][randColumn]
        g.setRed()
        i = random.randrange(0,9)
        self.drawFigure(randRow, randColumn, i)
        self.refresh(None)

    def drawFigure(self, row, column, image_label):
        g = self.gameboxs[row][column]
        image = PhotoImage(file="bmp/label_left_" + str(image_label) + ".png")
        g.setFigureImage(image)
        self.refresh(None)


    def refresh(self, event):
        '''Redraw the board, possibly in response to window being resized'''
        print("REFRESH")
        self.canvas.delete("all")
        if event != None:
            width = event.width - 1
            height = event.height - 1
        else:
            width = self.canvas.winfo_width() - 1
            height = self.canvas.winfo_height() - 1

        xsize = int((width) / self.columns)
        ysize = int((height) / self.rows)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        color = self.color2

        for row in range(self.rows):
            for col in range(self.columns):
                g = self.gameboxs[row][col]
                g.update(self.size)
                self.canvas.create_rectangle(g.x1, g.y1, g.x2, g.y2, outline="black", fill=g.currentColor,
                                             tags="square")
                if g.image is not None:
                    self.canvas.create_image((g.x1+g.size//2,g.y1+g.size//2), image=g.image)
                    self.canvas.pack()

    def update(self):
        ## TODO: uncomment print
        print("UPDATE: " + str(datetime.datetime.now()))
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.rightCam.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        ## TODO: Add code for mouth recognition here!

        self.parent.after(self.delay, self.update)


class GameBox:
    actualColor = ""
    currentColor = ""
    figure = None
    image = None
    figure_color = "#000000"
    row = 0
    column = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    size = 0
    label = False

    def __init__(self, row, column, size, color, label=False):
        self.size = size
        self.row = row
        self.column = column
        self.actualColor = color
        self.currentColor = color
        self.x1 = (column * self.size)
        self.y1 = (row * self.size)
        self.x2 = self.x1 + self.size
        self.y2 = self.y1 + self.size

        self.label = label

    def setColor(self, color):
        self.actualColor = color
        self.currentColor = color

    def setRed(self):
        self.currentColor = "red"

    def setFigureImage(self, image):
        self.image = image

    def update(self, size):
        self.size = size

        if self.column == 0:
            self.x1 = (self.column * self.size)#+20
            self.x2 = self.x1 + self.size
        else:
            self.x1 = (self.column * self.size)
            self.x2 = self.x1 + self.size

        if self.row == 0:
            self.y1 = (self.row * self.size)#+20
            self.y2 = self.y1 + self.size
        else:
            self.y1 = (self.row * self.size)
            self.y2 = self.y1 + self.size


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        # TODO: If needed manipulate here the size of vid window
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        ret, frame = self.vid.read()
        if self.vid.isOpened():
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    root = Tk()
    board = GameBoard(root)
    board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.resizable(False, False)
    root.mainloop()
