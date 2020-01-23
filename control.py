from tkinter import *
from UI import GameBoard
import lipnet_NN

class ControlUnit:

    def __init__(self):
        self.root = Tk()
        self.gameBoard = GameBoard(self.root)
        self.gameBoard.pack(side="top", fill="both", expand="true", padx=0, pady=4)

    def convert_ascii_to_column(self, sub_command):
        """converts ASCII in to the Value of the assigned column like A to 0"""

        return ord(sub_command.upper()) - 65

    def convert_in_number(self, com):
        """converts strings of numbers 0 to 8 like 'one' into the intvalue 1 """

        if com == "one":
            return 1
        elif com == "two":
            return 2
        elif com == "three":
            return 3
        elif com == "four":
            return 4
        elif com == "five":
            return 5
        elif com == "six":
            return 6
        elif com == "seven":
            return 7
        elif com == "eight":
            return 8
        elif com == "zero":
            return 0
        else:
            return com

    def controlfunction(self, commands_spoken):
        """ Method for updating the UI for net Output (commands_spoken)"""



        # Parts of the Command
        row = self.convert_ascii_to_column(commands_spoken[2])
        column = self.convert_in_number(commands_spoken[1])
        item = commands_spoken[3]
        color = commands_spoken[0]

        # Used the actual Command to fill the variables above with the corresponding values


        # set values in gameBoard and draw image
        if self.gameBoard is not None:
            self.gameBoard.drawFigure(row, column, 6)


    def mainloop(self):
        self.root.mainloop()



if __name__ == "__main__":
    newControlUnit = ControlUnit()
    # Methods for "testing"
    # use newControlUnit.controlfunction(net_output) for changes

    newControlUnit.mainloop()
