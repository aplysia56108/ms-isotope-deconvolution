import copy
import datetime
import json
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
import numpy as np
import os
import shutil
import sys
import time
import pandas as pd
from joblib import Parallel, delayed
settingJson = open(os.path.dirname(__file__) + '/setting.json', 'r')
settings = json.loads(settingJson.read())
settingJson.close()

# Open Matrix File
matrix_open = pd.read_csv(settings['matrix_TPD']['path'], skiprows = settings['matrix_TPD']['skiprows'])
matrix_nan = matrix_open.values
labels = matrix_open.columns
RawData_nan = pd.read_csv(settings['rawdata_TPD']['path'], skiprows = settings['rawdata_TPD']['skiprows']).values

#Remove nan raw and column:0
matrix_nan_1 = matrix_nan[:,~np.isnan(matrix_nan).all(axis=0)]
RawData_nan_1 = RawData_nan[:,~np.isnan(RawData_nan).all(axis=0)]
matrix_nan_2 = matrix_nan_1[~np.isnan(matrix_nan_1).all(axis=1)]
RawData_nan_2 = RawData_nan_1[~np.isnan(RawData_nan_1).all(axis=1)]
matrix_0 = np.nan_to_num(matrix_nan_2)
RawData_0 = np.nan_to_num(RawData_nan_2)
labels = labels[:matrix_0.shape[1]]

# Meet row and column number:1
if len(RawData_0.shape) == 1:
   RawData = np.reshape(RawData_0, (1, -1))
if matrix_0.shape[0] > RawData_0.shape[1]:
    RawDatacolumnnumber = RawData_0.shape[1]
    RawDatarownumber = RawData_0.shape[0]
    RawData = RawData_0
    matrix = np.delete(matrix_0, slice(RawDatacolumnnumber, matrix_0.shape[0]), axis = 0)
    
elif matrix_0.shape[0] < RawData_0.shape[1]:
    RawData = np.delete(RawData_0, slice(matrix_0.shape[0], RawData_0.shape[1]), axis = 1)
    RawDatacolumnnumber = RawData.shape[1]
    RawDatarownumber = RawData.shape[0]
    matrix = matrix_0

print(RawData.shape)

# Prepare for outputs
resultoutput = np.zeros([RawDatarownumber, matrix.shape[1]])
simoutput = np.zeros([RawDatarownumber, RawDatacolumnnumber])
logoutput = np.zeros([RawDatarownumber, 4])
starttime = time.time()

def f(x, matrix, data):
        simVec = matrix.dot(x)
        RMSE = np.linalg.norm(simVec-data, ord=2)
        return(RMSE**2)

def grad(x, matrix, data):
    simVec = matrix.dot(x)
    g = np.array([2*np.dot((simVec-data),(matrix[:,i])) for i in range(matrix.shape[1])])
    return(g)
    
# Run solver
def runSolver(Datanumber):
    lapstarttime = time.time()

    '''
    def hess(x):
        h = np.empty((0,matrix.shape[1]))
        for j in range (matrix.shape[1]):
            h_j = np.array([2*np.dot((matrix[:,j]-RawData[Datanumber]),(matrix[:,i])) for i in range(matrix.shape[1])])
            h = np.vstack((h,h_j))
        return(h)
    '''

    x0 = [0]* matrix.shape[1]

    result = minimize(fun = f,
                        x0 = x0, 
                        jac = grad,
                        args = (matrix, RawData[Datanumber]),
                        #hess = hess,
                        bounds = ((0,None),) *matrix.shape[1],
                        options = {'ftol':0,'gtol':0, 'maxfun':1000000, 'maxiter':1000000},
                        method = 'L-BFGS-B',
                        )
    resultout = result.x
    sim = matrix.dot(result.x)
    laptime = time.time() - lapstarttime
    log = np.array([Datanumber,result.success,result.fun,laptime])
    print(str(Datanumber + 1) + '/' + str(RawData.shape[0]) + ' Finished!! (success : ' + str(result.success) + ', n*(RMSE^2) : ' + str(result.fun) + ', time : ' + str(laptime) + ' s, step : ' + str(result.nfev) + ')')
    return Datanumber, resultout, sim, log

print('Start solving!')
processed = Parallel(n_jobs=8)([delayed(runSolver)(Datanumber) for Datanumber in range(RawDatarownumber)])
for i in processed:
    resultoutput[i[0]] = i[1]
    simoutput[i[0]] = i[2]
    logoutput[i[0]] = i[3]

# Result and log output
resultoutput = np.vstack((labels, resultoutput))
logoutput = np.vstack((np.array(['data number','success','n*(RMSE^2)','time']),logoutput))
resultoutput = np.hstack((logoutput[:, 1:2], resultoutput))
np.savetxt(settings['outputfilepath']['resultpath'], resultoutput, delimiter=',', fmt='%s')
np.savetxt(settings['outputfilepath']['simpath'], simoutput, delimiter=',', fmt='%s')
np.savetxt(settings['outputfilepath']['logpath'], logoutput, delimiter=',', fmt='%s')

# Make new folder
datatime_now = datetime.datetime.now()

new_dir_path = copy.deepcopy('./data/previous results/' + datatime_now.strftime('%Y%m%d_%H%M%S'))

os.mkdir(new_dir_path)

# Move data to the folder
shutil.move(settings['outputfilepath']['resultpath'] , new_dir_path)
shutil.move(settings['outputfilepath']['simpath'] , new_dir_path)
shutil.move(settings['outputfilepath']['logpath'] , new_dir_path)
shutil.copy(settings['matrix_TPD']['path'] , new_dir_path)
shutil.copy(settings['rawdata_TPD']['path'] , new_dir_path)

totaltime = time.time() - starttime
print('\n\n\n##################################')
print('Done! The Result was saved at : ' + new_dir_path + '.' + 'It took ' + str(totaltime) + ' s in total.' )
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.htmls
