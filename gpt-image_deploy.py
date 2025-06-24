import numpy
test = [1,2,3,4,5,6]
# test = 5.0
test = numpy.array(test)
A = test.astype('float32') /255
print(A)