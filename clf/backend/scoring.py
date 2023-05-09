class Scoring:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.TP = self._TP()
        self.FN = self._FN()
        self.FP = self._FP()

    def recall(self):
        #recall formula
        recall = self.TP/(self.TP+self.FN)
        # print(recall)
        return recall
     
    def precision(self):
        #precision formula
        precision = self.TP/(self.TP + self.FP)
        # print(precision)
        return precision

    def _TP(self):
        TP = 0
        for item in self.y_pred:
            if item in self.y_true:
                TP += 1
        return TP
    
    def _FN(self):
        FN = 0
        for item in self.y_true:
            if item not in self.y_pred:
                FN += 1
        # print('false neg', FN)
        return FN

    def _FP(self):
        FP = 0
        for item in self.y_pred:
            if item not in self.y_true:
                FP += 1
        # print('false pos', FP)
        return FP
    



