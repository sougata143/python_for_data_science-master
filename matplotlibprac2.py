import numpy as np
import matplotlib.pyplot as plt

p =np.random.standard_normal((50,2))
p += np.array((-1,1)) # center the distribution at (-1,1)

q =np.random.standard_normal((50,2))
q += np.array((1,1)) #center the distribution at (-1,1)


plt.scatter(p[:,0], p[:,1], color ='.25')
plt.scatter(q[:,0], q[:,1], color = '.75')
plt.show()

dd =np.random.standard_normal((50,2))
plt.scatter(dd[:,0], dd[:,1], color ='1.0', edgecolor ='0.0') # edge color controls the color of the edge
plt.show()

#Custom Color for Bar charts,Pie charts and box plots
print("Custom Color for Bar charts,Pie charts and box plots")
vals = np.random.random_integers(99, size =50)
color_set = ['.00', '.25', '.50','.75']
color_lists = [color_set[(len(color_set)* val) // 100] for val in vals]
c = plt.bar(np.arange(50), vals, color = color_lists)
plt.show()

hi =np.random.random_integers(8, size =10)
color_set =['.00', '.25', '.50', '.75']
plt.pie(hi, colors = color_set)# colors attribute accepts a range of values
plt.show()
#If there are less colors than values, then pyplot.pie() will simply cycle through the color list.
#  In the preceding 
#example, we gave a list of four colors to color a pie chart that consisted of eight values.
#  Thus, each color will be used twice


# values = np.random.randn(100)
# w = plt.boxplot(values)
# for att, lines in w.iteritems():
#     for l in lines:
#         l.set_color('k')
# plt.show()

#Colour Map
print("Colour Map")
# how to color scatter plots
#Colormaps are defined in the matplotib.cm module. This module provides 
#functions to create and use colormaps. It also provides an exhaustive choice of predefined color maps.
import matplotlib.cm as cm
N = 256
angle = np.linspace(0, 8 * 2 * np.pi, N)
radius = np.linspace(.5, 1., N)
X = radius * np.cos(angle)
Y = radius * np.sin(angle)
plt.scatter(X,Y, c=angle, cmap = cm.hsv)
plt.show()

#Color in bar graphs
# import matplotlib.cm as cm
# vals = np.random.random_integers(99, size =50)
# cmap = cm.ScalarMappable(col.Normalize(0,99), cm.binary)
# plt.bar(np.arange(len(vals)),vals, color =cmap.to_rgba(vals))
# plt.show()

#Line Styles
print("Line Styles")
# I am creating 3 levels of gray plots, with different line shades 


def pq(I, mu, sigma):
    a = 1. / (sigma * np.sqrt(2. * np.pi))
    b = -1. / (2. * sigma ** 2)
    return a * np.exp(b * (I - mu) ** 2)

I =np.linspace(-6,6, 1024)

plt.plot(I, pq(I, 0., 1.), color = 'k', linestyle ='solid')
plt.plot(I, pq(I, 0., .5), color = 'k', linestyle ='dashed')
plt.plot(I, pq(I, 0., .25), color = 'k', linestyle ='dashdot')
plt.show()

N = 15
A = np.random.random(N)
B= np.random.random(N)
X = np.arange(N)
plt.bar(X, A, color ='.75')
plt.bar(X, A+B , bottom = A, color ='W', linestyle ='dashed') # plot a bar graph
plt.show()

def gf(X, mu, sigma):
    a = 1. / (sigma * np.sqrt(2. * np.pi))
    b = -1. / (2. * sigma ** 2)
    return a * np.exp(b * (X - mu) ** 2)

X = np.linspace(-6, 6, 1024)
for i in range(64):
    samples = np.random.standard_normal(50)
    mu,sigma = np.mean(samples), np.std(samples)
    plt.plot(X, gf(X, mu, sigma), color = '.75', linewidth = .5)

plt.plot(X, gf(X, 0., 1.), color ='.00', linewidth = 3.)
plt.show()

#Fill surfaces with pattern
print("Fill surfaces with pattern")
N = 15
A = np.random.random(N)
B= np.random.random(N)
X = np.arange(N)
plt.bar(X, A, color ='w', hatch ='x')
plt.bar(X, A+B,bottom =A, color ='r', hatch ='/')
plt.show()
# some other hatch attributes are :
#/
#\
#|
#-
#+
#x
#o
#O
#.
#*


#Marker styles
print("Marker styles")
X= np.linspace(-6,6,1024)
Ya =np.sinc(X)

Yb = np.sinc(X) +1

plt.plot(X, Ya, marker ='o', color ='.75')
plt.plot(X, Yb, marker ='^', color='.00', markevery= 32)# this one marks every 32 nd element
plt.show()

# Marker Size
A = np.random.standard_normal((50,2))
A += np.array((-1,1))

B = np.random.standard_normal((50,2))
B += np.array((1, 1))

plt.scatter(A[:,0], A[:,1], color ='k', s =25.0)
plt.scatter(B[:,0], B[:,1], color ='g', s = 100.0) # size of the marker is specified using 's' attribute
plt.show()

# more about markers
X =np.linspace(-6,6, 1024)
Y =np.sinc(X)
plt.plot(X,Y, color ='r', marker ='o', markersize =9, markevery = 30, markerfacecolor='w', 
                linewidth = 3.0, markeredgecolor = 'b')
plt.show()


import matplotlib as mpl
mpl.rc('lines', linewidth =3)
mpl.rc('xtick', color ='w') # color of x axis numbers
mpl.rc('ytick', color = 'w') # color of y axis numbers
# mpl.rc('axes', facecolor ='g', edgecolor ='y') # color of axes 
mpl.rc('figure', facecolor ='.00',edgecolor ='w') # color of figure
# mpl.rc('axes', color_cycle = ('y','r')) # color of plots
x = np.linspace(0, 7, 1024)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()