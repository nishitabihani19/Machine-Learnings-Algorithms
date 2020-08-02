import matplotlib.pyplot as plt

x=[10,10,40,40,10],[20,20,30,30,20]
y=[10,40,40,10,10],[20,30,30,20,20]
plt.plot(x,label='x_axis')
plt.plot(y,label='y_axis')
#plt.legend()
plt.plot(x,y)
#plt.plot(x,y,'r.',linewidth=10,markersize=2,linestyle='dashdot')
plt.show()

