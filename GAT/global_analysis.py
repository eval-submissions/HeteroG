import os
import sys
import subprocess
prefix=int(sys.argv[1])

FNULL = open(os.devnull, 'w')
for i in range(1,prefix+1):
    folder = "data/graph"+str(i)
    p=subprocess.Popen("python test.py "+folder, stdout=FNULL, shell=True)
    p.communicate()
    p=subprocess.Popen("python anaysis_strategy.py "+folder+" >"+folder+"/anay.log", stdout=FNULL, shell=True)
    p.communicate()
    p=subprocess.Popen("python modify_test.py "+folder, stdout=FNULL, shell=True)
    p.communicate()