import numpy as np
from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, gamma

df_first = pd.read_csv("framingham.csv")
print("number of rows", len(df_first)) # number of rows
df = df_first.dropna()
print("number of rows without missing values", len(df), '\n----------------') # number of rows without missing values
print(df.columns)

def integrand(x, a_prop, scale_prop, a_, scale_):
    return gamma.pdf(x, a_prop, scale=scale_prop)*(np.sum(np.log(gamma.pdf(df['heartRate'].to_numpy(), a_, scale=scale_))) + np.log(gamma.pdf(x, a_, scale=scale_))) # Example function; replace with actual computation

# from scipy.optimize import minimize
def objective(a_prop, scale_prop, a_domain, scale_domain, sign=1): # make sure to pass sign=-1 for maximization
    # theta is [a, scale] for our Gamma we want to fit
    computed_integral = quad(integrand, 40, 145, args=(a_prop, scale_prop, a_domain, scale_domain))
    return sign*computed_integral[0], computed_integral[1] # returns [integral value, error estimate]

# a range of possible values a and scale can hold, upon which we run the optimization
a_star = arange(10, 80, 0.5) # a
scale_star = arange(0.5, 5.0, 0.1) # scale

# create a mesh from the axis
x, y = meshgrid(a_star, scale_star)

data = np.zeros((len(a_star)*len(scale_star), 3))
i = 0
initial_a, initial_s = 42, 1.8
for a in a_star:
    print('starting a new a', a)
    for s in scale_star:
        print(objective(initial_a, initial_s, a, s, sign=-1)[0])
        data[i] = [a, s, objective(initial_a, initial_s, a, s, sign=-1)[0]]
        i += 1

print("total combinations", i)

print(np.argmin(data[:,2], axis=0))
print(data[np.argmin(data[:,2], axis=0)])

# create a surface plot with the jet color scheme
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
# show the plot
plt.show()