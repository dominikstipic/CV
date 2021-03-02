import re
from Cell import Cell
import numpy as np

class Sheet:

    def __init__(self,rows,cols):
        assert rows > 0 and cols > 0, "row and column number must be positive number"
        self.cells = [[Cell(self,self._fromCoords(i,j)) for j in range(cols)] for i in range(rows)]
        self.observers = []
        # observeri su samo one ćelije koje ovise o drugima
    
    def cell(self,ref):
        """
        Returns referenced cell. For example A1 returns cell at position (0,0)
        """
        assert re.match("[A-Z][0-9]+",ref), "Illegal argument format"
        i,j = self._getCoordinates(ref)
        return self.cells[i][j]
         
    def set(self,ref,content):
        """
        Sets the cell content at indexed position.
        Content can be constant or other cell indexes
        """
        assert re.match("[A-Z][0-9]+",ref), "Illegal argument format"

        i,j = self._getCoordinates(ref)
        self.cells[i][j].update(content)
        self.evaluateSheet()
        # obavjesti ostale promatrače o promjeni
        
    def getrefs(self,cell):
        """
        Gets the list of all cells that given cell references
        """
        refs = re.findall("[A-Z][0-9]+", cell.exp)
        coords = map(lambda x : self._getCoordinates(x),refs)
        cells = map(lambda t : self.cells[t[0]][t[1]], coords)
        return list(cells)

    def _getCoordinates(self,ref):
        """
        Converts sheet style indexing into coordinate style indexing
        """
        groups = re.match("([A-Z])([0-9]+)",ref).groups()
        i = int(ord(groups[0]) - ord("A")) 
        j = int(groups[1]) - 1

        if (j >= len(self.cells) or i >= len(self.cells[-1])) : 
            raise IndexError("provided reference doesnt exist in sheet")

        return i,j
    
    def _fromCoords(self,i,j):
        i = chr(i + ord("A"))
        j = str(j+1)
        return i+j 

    # notifyAllObservers()
    def evaluateSheet(self):
        for o in self.observers:
            o.evaluate()
        for o in self.observers:
            o.evaluate()
        

    # removing observer
    def remove_observer(self,cell):
        self.observers = list(filter(lambda o : o != cell, self.observers))
        
    def printSheet(self):
        vec_str  = map(lambda x : x.exp , np.array(self.cells).reshape(-1).tolist())
        max_char_len = len(max(vec_str, key=lambda x : len(x)))
        for i in self.cells:
            for j in i:
                exp = j.exp
                if not exp:
                    exp = "*"
                pad = max_char_len - len(exp)
                leftpad  = " " * (pad//2)
                rightpad = " " * (pad - pad//2)
                print( "|"+ leftpad + exp + rightpad,end="")
            print("|")
        
        print("\n")

        
    





