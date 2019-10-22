#introduction of the basic functionalities of Matplotlib, the basic figure types

# Matlplotlib is a Python module for visualization. Matplotlib allows you to easily make 
# line graphs, pie chart, histogram and other professional grade figures. Using Matplotlib you can 
# customize every aspect of a figure. When used within IPython, Matplotlib has interactive features 
# like zooming and panning. It supports different GUI back ends on all operating systems, 
#and can also export graphics to common vector and graphics formats: PDF, SVG, JPG, PNG, BMP, GIF, etc.


import matplotlib.pyplot as plt #importing matplot lib library
import numpy as np 

x = range(100) 
print("x: {}".format(x))
y =[val**2 for val in x] 
print("y: {}".format(y))
plt.plot(x,y) #plotting x and y
plt.show()

#Using numpy
print("using numpy")
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x,y)
plt.show()

x= np.linspace(-3,2, 200)
Y = x ** 2 - 2 * x + 1.
plt.plot(x,Y)
plt.show()

# plotting multiple plots
# Matplot lib picks different colors for different plot. 
x =np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
z = np.cos(x)
plt.plot(x,y) 
plt.plot(x,z)
plt.show()

data = np.loadtxt('numpy.txt')
plt.plot(data[:,0], data[:,1]) # plotting column 1 vs column 2
plt.show()

data1 = np.loadtxt('scipy.txt') # load the file
print (data1.T)

for val in data1.T: #loop over each and every value in data1.T
    plt.plot(data1[:,0], val) #data1[:,0] is the first row in data1.T
plt.show()

#Scatter Plots and Bar Graphs
print("Scatter Plots and Bar Graphs")
sct = np.random.rand(20, 2)
print(sct)
plt.scatter(sct[:,0], sct[:,1]) # I am plotting a scatter plot.
plt.show()

ghj =[5, 10 ,15, 20, 25]
it =[ 1, 2, 3, 4, 5]
plt.bar(ghj, it) # simple bar graph
plt.show()

# you can change the thickness of a bar, by default the bar will have a thickness of 0.8 units
ghj =[5, 10 ,15, 20, 25]
it =[ 1, 2, 3, 4, 5]
plt.bar(ghj, it, width =5)
plt.show()

# barh is a horizontal bar graph
ghj =[5, 10 ,15, 20, 25]
it =[ 1, 2, 3, 4, 5]
plt.barh(ghj, it)
plt.show()

#Multiple bar charts
new_list = [[5., 25., 50., 20.], [4., 23., 51., 17.], [6., 22., 52., 19.]]
x = np.arange(4) 
plt.bar(x + 0.00, new_list[0], color ='b', width =0.25)
plt.bar(x + 0.25, new_list[1], color ='r', width =0.25)
plt.bar(x + 0.50, new_list[2], color ='g', width =0.25)
plt.show()

#Stacked Bar charts
p = [5., 30., 45., 22.]
q = [5., 25., 50., 20.]
x =range(4)
plt.bar(x, p, color ='b')
plt.bar(x, q, color ='y', bottom =p) 
plt.show()

# plotting more than 2 values
A = np.array([5., 30., 45., 22.])
B = np.array([5., 25., 50., 20.])
C = np.array([1., 20., 31., 1.])
X = np.arange(4)
plt.bar(X, A, color = 'b')
plt.bar(X, B, color = 'g', bottom = A)
plt.bar(X, C, color = 'r', bottom = A + B) # for the third argument, I use A+B
plt.show()

black_money = np.array([5., 30., 45., 22.]) 
white_money = np.array([5., 25., 50., 20.])
z = np.arange(4)
plt.barh(z, black_money, color ='g')
plt.barh(z, -white_money, color ='r')# - notation is needed for generating, back to back charts
plt.show()

#Pie Chart
print("Poe chart")
y = [5, 25, 45, 65]
plt.pie(y)
plt.show()

#histogram
print("histogram")
d = np.random.randn(100)
plt.hist(d, bins = 20)
plt.show()

#Box plot
print("boxplot")
d = np.random.randn(100)
plt.boxplot(d)
#1) The red bar is the median of the distribution
#2) The blue box includes 50 percent of the data from the lower quartile to the upper quartile. 
#   Thus, the box is centered on the median of the data.
plt.show()
d = np.random.randn(100, 5) # generating multiple box plots
plt.boxplot(d)
plt.show()