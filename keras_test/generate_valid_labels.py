import numpy as np
if __name__ == "__main__":
	validation_labels = np.array([1] * 298 + [2] * 265 + [3] * 121 + [4] * 49 + [5] * 239 + [6] * 17 + [7] * 5 + [8] * 7)
	np.savetxt('valid_labels.csv', validation_labels, fmt='%d', delimiter=",")