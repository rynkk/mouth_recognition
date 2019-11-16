from tkinter import *

import random


class GameBoard(Frame):
    gameboxs = ""

    def __init__(self, parent, rows=8, columns=8, size=50, color1="white", color2="black"):

        '''size is the size of a square, in pixels'''

        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        canvas_width = columns * size
        canvas_height = rows * size

        self.gameboxs = [[GameBox(i, j, 50, "white") for j in range(columns)] for i in range(rows)]

        color = self.color2

        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2

            for col in range(self.columns):
                self.gameboxs[row][col].setColor(color)
                color = self.color1 if color == self.color2 else self.color2

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
            color = self.color1 if color == self.color2 else self.color2

            for col in range(self.columns):
                self.gameboxs[row][col].setColor(color)
                color = self.color1 if color == self.color2 else self.color2

        randRow = random.randrange(0, 8)
        randColumn = random.randrange(0, 8)
        g = self.gameboxs[randRow][randColumn]
        g.setRed()
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
    row = 0
    column = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    size = 0

    def __init__(self, row, column, size, color):
        self.size = size
        self.row = row
        self.column = column
        self.actualColor = color
        self.currentColor = color
        self.x1 = (column * self.size)
        self.y1 = (row * self.size)
        self.x2 = self.x1 + self.size
        self.y2 = self.y1 + self.size

    def setColor(self, color):
        self.actualColor = color
        self.currentColor = color

    def setRed(self):
        self.currentColor = "red"

    def update(self, size):
        self.size = size
        self.x1 = (self.column * self.size)
        self.y1 = (self.row * self.size)
        self.x2 = self.x1 + self.size
        self.y2 = self.y1 + self.size


if __name__ == "__main__":
    root = Tk()
    board = GameBoard(root)
    board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.mainloop()
