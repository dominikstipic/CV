import re,ast

class Cell():
   

    def __init__(self, subject,name ,exp=""):
        self.sheet = subject
        # Subjekt 
        self.name  =  name
        # naziv Cella : A2,A1...
        self.exp   = exp
        # Izraz unutar Cella
        self.oper  = []
        # kompozicija operacija 
        self.value = []
        # Vrijednost 
        self.D = []

    def evaluate(self):

        rCells = self.sheet.getrefs(self)
        if not rCells:
            return

        for c in rCells:
            if not c.exp:
                continue
            elif self.is_constant(c.exp):
                self.D[c.name] = c.exp
            else:
                elem = c.evaluate()
                if elem:
                    self.D[c.name] = c.exp
                else:
                    return
        
        self.eval_expression()
                
    
    def eval_expression(self):
        exp_literal=re.sub(r'[A-Z][0-9]+',lambda m: self.D.get(m.group(0)), self.exp)
        self.exp   = exp_literal
        try:
            self.value = eval(exp_literal)
            self.exp   = str(self.value)
            self.sheet.remove_observer(self)
        except Exception:
            return

    def is_constant(self,content):
        return bool(re.match(r"[0-9]+",content))


        


    def update(self,content):
        """
        Sets the current cell operation and cell content
        """
        self.exp = content
        if self.is_constant(content):
            self.value = float(content)
        else:
            self.sheet.observers.append(self)
            # prijavi se kod subjekta -> Cell ovisi o drugim Cellovima
            variables = re.findall("[A-Z][0-9]+",content)
            self.D    = dict(zip(variables,variables))  
            






