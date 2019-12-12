from tkinter import *
from PIL import Image, ImageTk


import random


class GameBoard(Frame):

    gameboxs = ""

    def __init__(self, parent, rows=10, columns=10, size=50, color1="lightgray", color2="lightgray"):

        '''size is the size of a square, in pixels'''

        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        canvas_width = columns * sizek
        canvas_height = rows * size

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

        self.canvas = Canvas(self, borderwidth=0, highlightthickness=0,

                             width=canvas_width, height=canvas_height)

        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        # this binding will cause a refresh if the user interactively
        # changes the window size

        self.canvas.bind("<Configure>", self.refresh)
        self.button = Button(self, text="Click!", command=self.newColor)
        self.button.pack(side=LEFT)


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
        g.setFigure("info", "blue")
        self.refresh(None)

    def drawFigure(self, row, column, figure, color):
        g = self.gameboxs[row][column]
        g.setFigure(figure, color)
        self.refresh(None)


    def refresh(self, event):
        '''Redraw the board, possibly in response to window being resized'''

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
                if g.figure is not None:
                    self.canvas.create_bitmap(g.x1+g.size//2, g.y1+g.size//2, bitmap=g.figure)

                if g.image is not None:
                    self.canvas.create_image((g.x1+g.size//2,g.y1+g.size//2), image=g.image)
                    self.canvas.pack()


                # for row in range(self.rows):
                #     color = self.color1 if color == self.color2 else self.color2
                #     for col in range(self.columns):
                #         x1 = (col * self.size)
                #         y1 = (row * self.size)
                #         x2 = x1 + self.size
                #         y2 = y1 + self.size
                #         self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square"
                #         color = self.color1 if color == self.color2 else self.color2


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

    def setFigure(self, figure, color):
        self.figure = figure
        self.figure_color = color

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


if __name__ == "__main__":
    root = Tk()
    board = GameBoard(root)
    board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.mainloop()
