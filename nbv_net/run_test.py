import sys
import os
import numpy as np

name_of_model = str(sys.argv[1])
#max_iteration = int(sys.argv[2])
max_iteration = 10
if max_iteration <0:
    max_iteration = 1000
print('testing '+ name_of_model)
iteration = 0
while iteration<max_iteration:
    while os.path.isfile('./data/'+name_of_model+'_'+str(iteration)+'.txt')==False:
        pass
    os.system('python nbv_inference.py '+ name_of_model + ' '+str(iteration))
    f = open('./log/ready.txt','a')
    f.close()
    iteration += 1
print('testing '+ name_of_model + ' over.')
