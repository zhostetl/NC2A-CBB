import numpy as np
import matplotlib.pyplot as plt

def plot_bracket(xvals,yvals):
    ymean = []
    # print(f"xvals: {xvals}, yvals: {yvals}")
    for y in yvals:
        plt.plot(xvals, [y[0],y[0]], marker='o', markersize=5, color='blue')
        plt.plot(xvals, [y[1],y[1]], marker='o', markersize=5, color='blue')
        plt.plot([xvals[1],xvals[1]], [y[0],y[1]], marker='', color='grey')
        ymean.append(np.mean(y))
    
    if len(ymean) >1:
        new_ys = [(ymean[i], ymean[i+1]) for i in range(0, len(ymean), 2)]
    else:
        new_ys = [(ymean[0], ymean[0])]
    if xvals[0]>0:
        new_xs = [x-1 for x in xvals]
    else:
        new_xs = [x+1 for x in xvals]
    return new_xs, new_ys

# xs = [1,2]
# ys = [ (i, i+1) for i in range(1,16,2)]
# print(ys)

# x1,y1 = plot_bracket(xs,ys)

# x2,y2 = plot_bracket(x1,y1)

# x3,y3 = plot_bracket(x2,y2)

# x4,y4 = plot_bracket(x3,y3)

# x5,y5 = plot_bracket(x4,y4)
# print(f"x5: {x5}, y5: {y5}")

y_test = {
        'reg1':{'ys':[ (i, i+1) for i in range(1,16,2)],'xs':[10,9]},
        'reg2':{'ys':[ (i, i+1) for i in range(1,16,2)],'xs':[-10,-9]},
        'reg3':{'ys':[ (i, i+1) for i in range(30,46,2)],'xs':[10,9]},
        'reg4':{'ys':[ (i, i+1) for i in range(30,46,2)],'xs':[-10,-9]},
          }
for yy in y_test:
    ys = y_test[yy]['ys']
    xs = y_test[yy]['xs']
    
    while len(xs)>=2:
        x,y = plot_bracket(xs,ys)
        xs = x
        ys = y

        # print(f"xs: {xs}, ys: {ys}")
        if ys[0][0] == ys[0][1]: 
            plot_bracket(xs,ys)
            break

plt.show()