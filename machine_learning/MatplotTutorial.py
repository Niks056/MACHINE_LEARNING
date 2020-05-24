from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
plt.plot([1,2,3],[4,5,1])
plt.show()

x=[5,8,10]
y=[12,16,6]
plt.plot(x,y)

x2=[6,9,11]
y2=[6,15,7]

plt.plot(x,y,'g',label='line one',linewidth=2)
plt.plot(x2,y2,'c',label='line two',linewidth=2)
plt.title('Epic info')
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.legend()
plt.grid(True,color='K')
plt.show()

plt.bar([1,3,5,7,9],[5,2,7,8,2],label='example One')
plt.bar([2,4,6,8,10],[8,6,2,5,6],label='Example Two',color='g')

population_ages=[22,55,62,45,21,22,34,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages,bins,histtype='bar',rwidth=0.8)


x=[1,2,3,4,5,6,7,8]
y=[5,2,4,2,1,4,5,2]

plt.scatter(x,y,label='skitscat',color='k')

days=[1,2,3,4,5]

sleeping=[7,8,6,11,7]
eating=[2,3,4,3,2]
working=[7,8,7,2,2]
playing=[8,5,7,8,13]

plt.plot([],[],color='m',label='Sleeping',linewidth=5)
plt.plot([],[],color='c',label='Eating',linewidth=5)
plt.plot([],[],color='r',label='Working',linewidth=5)
plt.plot([],[],color='k',label='Playing',linewidth=5)

plt.stackplot(days,sleeping,eating,working,playing,colors=['m','c','r','k'])

slices=[7,2,2,13]
activities=['Sleeping','eating','working','playing']
cols=['c','m','r','b']


plt.pie(slices,labels=activities,colors=cols,startangle=90,shadow=True,explode=(0,0.1,0,0),autopct='%1.1f%%')
plt.title('Pie Plot')
plt.show()

#explode helps to highlight the part
#autopct heps to add % in pie_charts.

import numpy as np

def f(t):
    return np.exp(-t) *np.cos(2*np.pi*t)

t1=np.arange(0.0,5.0,0.1)
t2=np.arange(0.0,5.0,0.02)

plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2))

plt.subplot(212)

#(211-- there are two graphs 1 means there are 2 horizontal graphs and 1 means this first graph)

plt.plot(t2,np.cos(2*np.pi*t2))
plt.show()