__author__ = 'Prateek'
import Classification


print "1 feature"
Classification.LogReg(2,3)
Classification.SVM(2,3)
Classification.DTC(2,3)

print "9 features"
Classification.LogReg(2,11)
Classification.SVM(2,11)
Classification.DTC(2,11)

print "16 features"
Classification.LogReg(2,18)
Classification.SVM(2,18)
Classification.DTC(2,18)

print "30 features"
Classification.LogReg(2,32)
Classification.SVM(2,32)
Classification.DTC(2,32)
