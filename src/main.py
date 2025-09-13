import sys
from Model import Applicant

def main():
    nameInput = sys.argv[1]
    ageInput = sys.argv[2]
    paySlipFile = sys.argv[3]
    billPayments = sys.argv[4]
    applicant =  Applicant(nameInput, ageInput, paySlipFile, billPayments)
    print(applicant._modelFeat)
