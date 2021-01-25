import math
import random
import matplotlib.pyplot as plt
import numpy as np


# Integration Simpson rule..
def simphson_integration(a, b, f, N=1000):
    ''' This present function is about simphson integration.'''
    inital_x = a  # inital boundary point
    final_x = b  # final boundary point
    dx = (final_x - inital_x) / N  # generating dx point
    sum = 0
    for i in range(N):
        # integration process

        if i == 0:
            sum += f(inital_x)
            inital_x += dx
        elif (i % 2) != 0:  # i is even
            sum += 4 * f(inital_x)
            inital_x += dx
        elif (i % 2) == 0:  # i is odd
            sum += 2 * f(inital_x)
            inital_x += dx
        if i == N - 1:
            sum += f(inital_x)

    return (dx / 3) * sum


# Planewave function
def wavefunc(k, x):
    value = math.e ** (1j * k * x)
    return float(value.real)


# 1st order Gradient of Function
def gradient1(k, func, r):
    h = 0.0001
    return (func(k, r + h) - func(k, r)) / h


# 2nd order Gradient of Function
def gradient2(k, func, r):
    h = 0.0001
    return ((gradient1(k, func, r + h)) - gradient1(k, func, r)) / h


# Sum of plane wave wavefunctions.
def sum_wavefunc(K, x):
    ''' k is an array in this problem'''
    wavesum = 0
    for i in range(len(K)):
        wavesum = wavesum + wavefunc(K[i], x)

    return wavesum


def gradient_sum_wave1(K, sum_func, r):
    h = 0.0001
    return (sum_func(K, r + h) - sum_func(K, r)) / h


def gradient_sum_wave2(K, sum_func, r):
    h = 0.0001
    return (gradient_sum_wave1(K, sum_func, r + h) - gradient_sum_wave1(K, sum_func, r)) / h


# Boundary energy potential
def V1(x):
    '''Simple box potential'''
    L = 10
    if x >= 0 and x <= L:
        V = -5
    else:
        V = 0
    return V


def E_known_energy(x):
    '''Simple box potential'''
    L = 10
    if x >= 0 and x <= L:
        E = -2
    else:
        E = 0
    return E


def E_integration_over_wave(x1, x2, K, V, sum_func):
    h = 1
    m = 1
    E_known_energy=0.314
    # simulation box
    r = [x1]
    small = 0.01
    while x1 < x2:
        x1 = x1 + small
        r.append(x1)

    gra2 = []
    func = []
    E_error_sum = 0
    for i in r:
        a = sum_func(K, i)
        func.append(a)
        b = gradient_sum_wave2(K, sum_func, float(i))
        gra2.append(b)
        k2 = b / a
        E_total_energy = -(h ** 2 * k2) / (2 * m) + V(i)
        E_error_sum = E_error_sum + abs(E_total_energy - E_known_energy)

    return E_error_sum


def generate_sample(x, sample_N=10):
    sample_set = []
    for k in range(sample_N):
        x_set = []
        for i in range(len(x)):
            random_x = random.uniform(-1, 1) * 0.01
            x_set.append(x[i] + random_x)
        sample_set.append(x_set)

    return sample_set


def search_algo():
    simul_bound_init = -5
    simul_bound_fnl = 15
    K = []
    for i in range(10):
        K.append(random.uniform(0.5, 0.6))

    while E_integration_over_wave(-0.5, 0.5, K, V1, sum_wavefunc) >= 0.1:
        #filehandle = open('wavefunc_param_new2.txt', 'a')

        mutated = generate_sample(K)
        COST = []
        for j in mutated:
            COST.append(E_integration_over_wave(-0.5, 0.5, j, V1, sum_wavefunc))

        min_cost = min(COST)
        print(min_cost)

        index_min_cost = COST.index(min_cost)
        final_K = mutated[index_min_cost]

        K = list(final_K)

        cost_line = 'cost: ' + str(min_cost) + '\n'
        #filehandle.write(cost_line)
        k_data = 'k_i=' + str(K) + '\n'
        #filehandle.write(k_data)
        #filehandle.close()
        print(K)

    return K

search_algo()


k_i1=[0.33974367709506353, 0.5015052550475873, 0.9019398014493342, 0.6788195610151461, 1.1295955061305067]
k_i=[0.7554899216729795, 0.7862935811161916, 0.806541360845643, 0.8022213214953788, 0.8107121544591318]
k_i11=[0.9913468628654323, 0.7623854149274232, 0.9264695806573522, 0.23776143936329017, 0.5885938062567493]
k_i00=[0.8145940062782712, 0.8079348577858096, 0.8119192949356293, 0.7194336670313155, 0.8073519739680358]

x_data=np.arange(-10,10,0.001)
y_data=[]
y2_data=[]
V_data=[]
for j in x_data:
    y_data.append(sum_wavefunc(k_i,j))
    y2_data.append(sum_wavefunc(k_i1,j))
    V_data.append(V1(j))

plt.plot(x_data,y_data)
plt.plot(x_data,y2_data)
#plt.plot(x_data,V_data)
plt.show()


