import os

f = open('411a3/val.csv')
y = [(line.split(','))[1] for line in f]

y = np.asarray(y)[1:].astype(np.int32) # starts at 1

np.save('val_labels', y)