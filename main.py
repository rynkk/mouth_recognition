from tkinter import *
from UI import GameBoard


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

    def controlfunction(self, commands_spoken, commands=None):
        """ Method for updating the UI for net Output (commands_spoken)"""

        if commands is None:
            # commands for standard Net_Output
            commands = ['blue', 'green', 'red', 'white',
                        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']

        index_of_spoken_elements = []
        # extracts the greatest Element of a section
        greatest_element = max(commands_spoken[0:4])
        index_of_spoken_elements.append(commands_spoken[0:4].index(greatest_element))
        greatest_element = (max(commands_spoken[4:13]))
        index_of_spoken_elements.append(commands_spoken[4:13].index(greatest_element)+4)

        greatest_element = max(commands_spoken[13:22])
        index_of_spoken_elements.append(commands_spoken[13:22].index(greatest_element)+12)
        greatest_element = max(commands_spoken[22:31])
        index_of_spoken_elements.append(commands_spoken[22:31].index(greatest_element)+21)

        # adds comments with index from above to an new List, that represents the new actual_command
        actual_command = []

        for i in index_of_spoken_elements:
            actual_command.append(commands[i])


        # Parts of the Command
        row = -1
        column = -1
        item = ""
        color = ""

        # Used the actual Command to fill the variables above with the corresponding values
        for com in actual_command:

            com = self.convert_in_number(com)

            if type(com) is int:
                row = com
            elif com == "green" or com == "blue" or com == "white" or com == "red":
                color = com
            elif len(com) == 1:
                if ord(com.upper()) in range(65, 65 + 9):
                    column = self.convert_ascii_to_column(com)
                elif ord(com.upper()) in range(65 + 9, 65 + 33):
                    item = self.convert_ascii_to_column(com)-9  # temporary solution until the correct images were added

        # set values in gameBoard and draw image
        if self.gameBoard is not None:
            self.gameBoard.drawFigure(row, column, item)

        return actual_command

    def mainloop(self):
        self.root.mainloop()

    def UI_test(self):
        for i in range(1, 4):
            self.gameBoard.drawFigure(i, i, i)

    def test(self):
        test = self.controlfunction([0, 0, 0, 1,
                                     0, 0.09, 1, 0, 0.9, 0, 0.5, 0, 0, 0,
                                     0, 0.09, 0, 0, 0, 0.87, 0, 0.6, 0,
                                     0, 0.09, 0, 0, 0.9, 0, 0.5, 0, 0])


if __name__ == "__main__":
    newControlUnit = ControlUnit()

    # Methods for "testing"
    newControlUnit.test()
    newControlUnit.UI_test()

    # use newControlUnit.controlfunction(net_output) for changes

    newControlUnit.mainloop()
