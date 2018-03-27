from numpy import *
import matplotlib.image as mpimg
import operator

pic = mpimg.imread('9.jpg')
mat_data = zeros((32,32))
fopen = open('cha.txt','w+')

for i in range(0,32):
	for j in range(0,32):
		mat_data[i][j] = int(pic[i][j])
		if(mat_data[i][j] <= 100):
			mat_data[i][j] = 1
		else:
			mat_data[i][j] = 0
		fopen.write(str(int(mat_data[i][j])))
	fopen.write('\n')
fopen.close()

print("finised...")