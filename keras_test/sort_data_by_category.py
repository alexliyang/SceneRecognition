import os
import numpy as np

f = open('train.csv')
y = [(line.split(','))[1] for line in f]

y = np.asarray(y)[1:].astype(np.int32) # starts at 1

for i in range (0, y.shape[0]-1):
    #os.rename("path/to/current/file.foo", "path/to/new/desination/for/file.foo")
    # >>> n = '4'
    # >>> print n.zfill(3)
    # 004

    current_file = str(i+1).zfill(5) + '.jpg' # Deal with the 0 indexing vs 1 indexing
    current_directory = 'train/' + current_file
    
    if (y[i] == 1):        
        new_directory = 'train/1-structures/' + current_file
    elif (y[i] == 2):        
        new_directory = 'train/2-indoor/' + current_file
    elif (y[i] == 3):        
        new_directory = 'train/3-people/' + current_file
    elif (y[i] == 4):        
        new_directory = 'train/4-animals/' + current_file
    elif (y[i] == 5):        
        new_directory = 'train/5-plantlife/' + current_file
    elif (y[i] == 6):        
        new_directory = 'train/6-food/' + current_file
    elif (y[i] == 7):        
        new_directory = 'train/7-car/' + current_file
    elif (y[i] == 8):        
        new_directory = 'train/8-sea/' + current_file
      
    os.rename(current_directory, new_directory)
    print('Moved from', current_directory, ' to ', new_directory)
