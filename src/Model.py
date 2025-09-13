

class Applicant:

    def __init__(self, appName, appAge, appIncome, appUtilities) -> None:
        self._name = appName
        self._age = appAge
        self._userIncome = appIncome
        self._utilityBill = appUtilities
        self._modelFeat = [self._age, self._userIncome, self._utilityBill]


    #def riskEngine(self):


    #def modelScore(self):
    #    return
        
