import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv

import gymnasium as gym

class SymbolicRegression:
    def __init__(self,domains, npoints, ndimensions):
        self.type='symbolicRegression'
        self.domains = domains
        self.ndimensions = ndimensions
        self.npoints = int(npoints * 0.8)
        self.npoints_test = int(npoints * 0.2)
        self.points = self.preds = self.targets = None
        self.points_test = self.preds_test = self.targets_test = None
        self.variables = ['var_' + str(i) for i in range(ndimensions)]

        self.initialize()

    def initialize(self):
        self.initializePreds()
        self.initializePoints()
    
    def initializePreds(self):
        self.preds = np.zeros((self.npoints, self.ndimensions))
        self.preds_test = np.zeros((self.npoints_test, self.ndimensions))

    def initializePoints(self):
        self.points = [[random.random() for j in range(self.ndimensions)] for i in range(self.npoints)]
        self.points_test = [[random.random() for j in range(self.ndimensions)] for i in range(self.npoints_test)]
        self.points.sort()
        self.points_test.sort()
        self.points = np.array(self.points)
        self.points_test = np.array(self.points_test)
        
        for i in range(self.npoints):
            for j in range(self.ndimensions):
                self.points[i][j] = self.points[i][j]*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0]
        self.targets = self.points
        
        for i in range(self.npoints_test):
            for j in range(self.ndimensions):
                self.points_test[i][j] = self.points_test[i][j]*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0]
        self.targets_test = self.points_test

    def evaluate(self, ind):
        for i in range(self.npoints):
            self.preds[i] = ind.arithmetic_interpret_mat(None, self.points[i], logExecutedNodes=True)
        
        val = self.bounded_MSE(self.preds, self.targets)
        if(np.isnan(val)):
            return np.inf
        return val
    
    def evaluate_test(self, ind):
        for i in range(self.npoints_test):
            self.preds_test[i] = ind.arithmetic_interpret_mat(None, self.points_test[i], logExecutedNodes=True)
        
        val = self.bounded_MSE(self.preds_test, self.targets_test)
        if(np.isnan(val)):
            return np.inf
        return val

    def RMSE(self, preds, targets):
        return np.sqrt(np.mean((preds - targets) ** 2))

    def MSE(self, preds, targets):
        return np.mean((preds - targets) ** 2)
    
    def bounded_MSE(self, preds, targets):
        return 1 - 1 / (1 + self.MSE(preds, targets))

class Koza1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**4 + points**3+points**2+points)[0]

class Koza2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**5 - 2*points**3 + points)[0]

class Koza3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**6 - 2*points**4 + points**2)[0]

class Nguyen1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**3 + points**2 + points)[0]

class Nguyen3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**5 + points**4 + points**3 + points**2 + points)[0]

class Nguyen4(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**6 + points**5 + points**4 + points**3 + points**2 + points)[0]


class Nguyen5(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (np.sin(points**2) * np.cos(points) - 1)[0]

class Nguyen6(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (math.sin(points) * math.sin(points+points**2))[0]

class Nguyen7(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (math.log(points+1) * math.log(points**2+1))[0]

class Nguyen8(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (np.sqrt(points))[0]

class Nguyen9(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return math.sin(points[0]) + math.sin(points[1]**2)

class Nguyen10(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 2*math.sin(points[0]) + math.cos(points[1])

class Nguyen11(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return points[0]**points[1]

class Nguyen12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return points[0]**4 - points[0]**3 + 0.5*points[1]**2 - points[1]


class Paige1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y = points[0], points[1]
        return 1.0/(1.0+x**-4) + 1.0/(1.0+y**-4)

class Korns1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)
    def initializePoints(self):
        self.domains = [[-50,50] for i in range(5)]
        self.npoints = 10000
        self.ndimensions=5
        self.points = [[random.random()*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0] for j in range(self.ndimensions)] for i in range(self.npoints)]

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y, z, w, v = points[0], points[1], points[2], points[3],points[4]
        return 1.0/(1.0+x**-4) + 1.0/(1.0+y**-4)


class Keijzer12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y  = points[0], points[1]
        return x**4 -x**3 + (y**2)/2 -y 


#the functions from this point onwards are not part of "GP needs better benchmarks"
class R1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return ((x+1.0)**3)/(x**2 - x + 1.0 )

class R2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)
    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return (x**5 - 3*x**3 + 1.0) / (x**2 + 1)

class R3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)
    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return (x**6 + x**5) / (x**4 + x**3 + x**2 + x + 1)

class Livemore1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return 1.0/3 + x + math.sin(x**2)

class Livemore2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return math.sin(x**2)*math.cos(x)-2.0

class Livemore3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return math.sin(x**3)*math.cos(x**2)-1.0

class Livemore4(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.log10(x+1) + np.log10(x**2+1) + np.log10(x)


class Livemore5(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return x**4 - x**3 + x**2 - y


class Livemore6(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return 4*x**4 + 3*x**3 + 2*x**2 + x

class Livemore7(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.sinh(x)

class Livemore8(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.cosh(x)

class Livemore9(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x

class Livemore10(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return 6 * np.sin(x) *np.cos(y)

class Livemore11(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return (x**2 * x**2) /(x+y)

class Livemore12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return (x**5) /(y**3)

class Livemore13(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**(1.0/3)

class Livemore14(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**3 + x**2 +x + np.sin(x) + np.sin(x**2)

class Livemore15(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**(1.0/5)

class Livemore16(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**(2.0/5)

class Livemore17(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return 4*np.sin(x)*np.cos(y)

class Livemore18(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.sin(x**2)*np.cos(x)-5

class Livemore19(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**5 + x**4 + x**2 + x

class Livemore20(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.exp(-x**2)

class Livemore21(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x


class Livemore22(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.exp(-0.5*x**2)

class Schwefel(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 418.9829*self.ndimensions - np.sum(points * np.sin(np.sqrt(abs(points))))

class Sphere(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return np.sum(points **2)    

class Rosenbrock(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        a = 0.0
        for i in range(len(points)-1):
            a+=100*(points[i+1]-points[i]**2)**2 + (points[i] -1)**2

class Rastringin(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 10*len(points) + np.sum(points**2 -10*np.cos(2*math.pi*points))    

class ArtificialAnt:
    def __init__(self, map_name):
        self.type='artificialAnt'
        self.x, self.y, self.heading = 0, 0, 'r'

        self.eaten=0
        self.map_name = map_name
        self.read_grid(map_name)

        self.total_food = np.sum(self.original_grid)
        self.grid = np.zeros(self.original_grid.shape)
        self.grid += self.original_grid

    def initialize(self):
        pass
        
    def read_grid(self,fname='sf_map.txt'):

        f = open(fname)
        lines = f.readlines()
        for i in range(len(lines)):
            if('\n' in lines[i]):
                lines[i] = lines[i][:lines[i].index('\n')]
        self.original_grid = np.zeros((len(lines[0]), len(lines)))
        
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                self.original_grid[i][j] = int(lines[i][j])
        f.close()

    def left(self):
        if(self.heading=='r'):
            self.heading = 'u'
        elif(self.heading == 'u'):
            self.heading = 'l'
        elif(self.heading == 'l'):
            self.heading = 'd'
        elif(self.heading =='d'):
            self.heading = 'r'
        return True
    
    def right(self):
        if(self.heading == 'l'):
            self.heading = 'u'
        elif(self.heading=='u'):
            self.heading = 'r'
        elif(self.heading =='r'):
            self.heading = 'd'        
        elif(self.heading == 'd'):
            self.heading = 'l'
        return True

    def move(self):
        if(self.heading == 'u'):
            self.x = (self.x-1)%self.grid.shape[0]
        elif(self.heading == 'd'):
            self.x = (self.x+1)%self.grid.shape[0]
        elif(self.heading == 'l'):
            self.y = (self.y-1)%self.grid.shape[1]
        elif(self.heading == 'r'):
            self.y = (self.y+1)%self.grid.shape[1]

        if(self.grid[self.x][self.y] == 1):
            self.grid[self.x][self.y] = 0
            self.eaten +=1

        return True

    def iffoodAhead(self):
        if(self.heading=='l'):
            x, y = self.x, (self.y-1)%self.grid.shape[1]
        elif(self.heading=='r'):
            x, y = self.x, (self.y+1)%self.grid.shape[1]
        elif(self.heading=='d'):
            x, y = (self.x+1)%self.grid.shape[0], self.y
        elif(self.heading=='u'):
            x, y = (self.x-1)%self.grid.shape[0], self.y
        return (self.grid[x][y] == 1)

    def evaluate(self,ind):
        self.eaten = 0
        self.grid = self.grid*0+self.original_grid
        ind.behaviour = np.zeros((2,self.steps))
        current_node = None
        c = 0
        for i in range(self.steps):            
            current_node, motion_terminated = ind.stateful_interpret_ma(self, current_node, logExecutedNodes=True)
            ind.behaviour[0][i] = self.x
            ind.behaviour[1][i] = self.y
            c=i
        while(c<self.steps):
            ind.behaviour[0][c] = ind.behaviour[0][c-1]
            ind.behaviour[1][c] = ind.behaviour[1][c - 1]
            c+=1
        return self.eaten
        
class SantaFeAntTrail(ArtificialAnt):
    def __init__(self,map_name='maps/sf_map.txt'):
        ArtificialAnt.__init__(self, map_name)
        self.steps = 600

class LosAltosHillsAntTrail(ArtificialAnt):
    def __init__(self,map_name='maps/lah_map.txt'):
        ArtificialAnt.__init__(self, map_name)
        self.steps = 3000

class SantaFeAntTrailMin(ArtificialAnt):
    def __init__(self,map_name='maps/sf_map.txt'):
        ArtificialAnt.__init__(self, map_name)
        self.steps = 600

    def evaluate(self,ind):
        return 89-super().evaluate(ind)

class LosAltosHillsAntTrailMin(ArtificialAnt):
    def __init__(self,map_name='maps/lah_map.txt'):
        ArtificialAnt.__init__(self, map_name)
        self.steps = 3000

    def evaluate(self,ind):
        return 157-super().evaluate(ind)

# 2025 Multiplexer
class BooleanMultiplexer:
    def __init__(self,domains, npoints, ndimensions):
        self.type='booleanMultiplexer'
        self.domains = domains
        self.ndimensions = ndimensions
        self.npoints = int(npoints * 0.8)
        self.npoints_test = int(npoints * 0.2)
        self.points = self.preds = self.targets = None
        self.points_test = self.preds_test = self.targets_test = None
        self.variables = ['bit_' + str(i) for i in range(ndimensions)]

        self.initialize()
     
    def initialize(self):
        self.initializePreds()
        self.initializePoints()
        self.initializeTargets()
        
    def initializePreds(self):
        self.preds = np.zeros(self.npoints)
        self.preds_test = np.zeros(self.npoints_test)
    
    def initializePoints(self):
        self.points = [[random.random() for j in range(self.ndimensions)] for i in range(self.npoints)]
        self.points_test = [[random.random() for j in range(self.ndimensions)] for i in range(self.npoints_test)]
        self.points.sort()
        self.points_test.sort()
        self.points = np.array(self.points)
        self.points_test = np.array(self.points_test)
        
        for i in range(self.npoints):
            for j in range(self.ndimensions):
                self.points[i][j] = self.points[i][j]*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0]
        self.targets = self.points
        
        for i in range(self.npoints_test):
            for j in range(self.ndimensions):
                self.points_test[i][j] = self.points_test[i][j]*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0]
        self.targets_test = self.points_test
        
    def evaluate(self, ind):
        for i in range(self.npoints):
            self.preds[i] = ind.boolean_interpret_mat(None, self.points[i], logExecutedNodes=True)

        val = self.MSE(self.preds, self.targets)
        if(np.isnan(val)):
            return np.inf
        return val
    
    def evaluate_test(self, ind):
        for i in range(self.npoints_test):
            self.preds_test[i] = ind.boolean_interpret_mat(None, self.points_test[i], logExecutedNodes=True)

        val = self.MSE(self.preds_test, self.targets_test)
        if(np.isnan(val)):
            return np.inf
        return val
        
    def RMSE(self, preds, targets):
        return np.sqrt(np.mean((preds - targets) ** 2))

    def MSE(self, preds, targets):
        return np.mean((preds - targets) ** 2)

class NMultiplexer3(BooleanMultiplexer):
    def __init__(self, domain, npoints, ndimensions):
        BooleanMultiplexer.__init__(self, domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])
        self.targets_test = np.array([self.func(self.points_test[i]) for i in range(self.npoints_test)])

    def func(self, points):
        s = int(round(points[0]))
        data_bits = [int(round(points[1])), int(round(points[2]))]
        return data_bits[s]

class NMultiplexer6(BooleanMultiplexer):
    def __init__(self, domain, npoints, ndimensions):
        BooleanMultiplexer.__init__(self, domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])
        self.targets_test = np.array([self.func(self.points_test[i]) for i in range(self.npoints_test)])

    def func(self, points):
        s0 = int(round(points[0]))
        s1 = int(round(points[1]))
        index = 2 * s1 + s0
        data_bits = [int(round(b)) for b in points[2:6]]
        return data_bits[index]
    
class NMultiplexer11(BooleanMultiplexer):
    def __init__(self, domain, npoints, ndimensions):
        BooleanMultiplexer.__init__(self, domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])
        self.targets_test = np.array([self.func(self.points_test[i]) for i in range(self.npoints_test)])

    def func(self, points):
        s0 = int(round(points[0]))
        s1 = int(round(points[1]))
        s2 = int(round(points[2]))
        index = s0 + 2*s1 + 4*s2
        data_bits = [int(round(b)) for b in points[3:11]]
        return data_bits[index]

class NMultiplexer20(BooleanMultiplexer):
    def __init__(self, domain, npoints, ndimensions):
        BooleanMultiplexer.__init__(self, domain, npoints, ndimensions)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])
        self.targets_test = np.array([self.func(self.points_test[i]) for i in range(self.npoints_test)])

    def func(self, points):
        s0 = int(round(points[0]))
        s1 = int(round(points[1]))
        s2 = int(round(points[2]))
        s3 = int(round(points[3]))
        index = s0 + 2*s1 + 4*s2 + 8*s3
        data_bits = [int(round(b)) for b in points[4:20]]
        return data_bits[index]

""" 
    In these problems, selection bits determine which of the data bits is chosen as the output.
    For example, in the 6-Multiplexer, the first 2 bits form an index (0 to 3) that selects one of the following 4 data bits. 
    These are deterministic problems that evaluate the ability of GP algorithms to manipulate boolean functions. 
    
    References: 
        Koza, J.R. (1992). Genetic Programming as a Means for Programming Computers by Natural Selection. 
        McDermott, J.R., et al. (2012). Genetic Programming Needs Better Benchmarks.
"""

class Spambase:
    def __init__(self, domains, npoints, ndimensions):
        self.type = 'Spambase'
        self.domains = domains
        self.ndimensions = ndimensions
        self.npoints = int(npoints * 0.8)
        self.npoints_test = int(npoints * 0.2)

        self.points = self.preds = self.targets = None
        self.points_test = self.preds_test = self.targets_test = None
        
        self.variables = ['var_' + str(i) for i in range(ndimensions)]

        self.initialize()

    def initialize(self):
        self.initializePreds()
        self.initializePoints()
    
    def initializePreds(self):
        self.preds = np.zeros(self.npoints)
        self.preds_test = np.zeros(self.npoints_test)
    
    def initializePoints(self):
        with open("spambase.csv") as spambase:
            spamReader = csv.reader(spambase)
            spam = list(list(float(elem) for elem in row) for row in spamReader)

        random.shuffle(spam)

        self.points = []
        self.targets = []
        for i in range(self.npoints):
            self.points.append(spam[i][:-1])
            self.targets.append(spam[i][-1])
        
        self.points_test = []
        self.targets_test = []
        for i in range(self.npoints, self.npoints + self.npoints_test):
            self.points_test.append(spam[i][:-1])
            self.targets_test.append(spam[i][-1])
        
        self.points = np.array(self.points)
        self.targets = np.array(self.targets) 
        
        self.points_test = np.array(self.points_test)
        self.targets_test = np.array(self.targets_test)    

    def evaluate(self, ind):
        for i in range(self.npoints):
            self.preds[i] = ind.spambase_interpret_mat(None, self.points[i], logExecutedNodes=True)
        
        f1_score = self.F1_score(self.preds, self.targets, self.npoints)
        
        return 100.0 - (f1_score * 100.0) # Usar f1 score percentual e com proposito de minimização como métrica de fitness.

    def evaluate_test(self, ind):
        for i in range(self.npoints_test):
            self.preds_test[i] = ind.spambase_interpret_mat(None, self.points_test[i], logExecutedNodes=True)
        
        f1_score = self.F1_score(self.preds_test, self.targets_test, self.npoints_test)

        return 100.0 - (f1_score * 100.0)

    def F1_score(self, preds, targets, npoints):
        TP = TN = FP = FN = 0
        for i in range(npoints):
            if preds[i] == targets[i]:
                if preds[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if preds[i] == 1:
                    FP += 1
                else:
                    FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score  = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        
        return f1_score

class Spambase1(Spambase):
    def __init__(self, domain, npoints, ndimensions,):
        super().__init__(domain, npoints, ndimensions)
        
class LunarLander:
    def __init__(self, domains, npoints, ndimensions):
        self.type = 'LunarLander'
        # self.env = gym.make("LunarLander-v3", render_mode = None, continuous = False, gravity = -10.0, enable_wind = True, wind_power = 15.0, turbulence_power = 1.5)
        self.env = gym.make("LunarLander-v3", render_mode = None, continuous = False, gravity = -10.0, enable_wind = False, wind_power = 0.0, turbulence_power = 0.0)
        self.num_episodes = 5
        self.max_steps = 1000
        
    def initialize(self):
        pass
        
    def evaluate(self, ind):
        total_reward = 0
        seeds = [None] * self.num_episodes  
        for i in range(self.num_episodes):
            state, _ = self.env.reset(seed=seeds[i])
            episode_reward = 0
            for _ in range(self.max_steps):
                action = ind.LunarLander_interpret_mat(None, state, logExecutedNodes=True)           
                if not isinstance(action, str) or '-action' not in action:
                    action = 0
                    episode_reward += -105.0  # minimum reward in LunarLander
                    break 
                elif 'left-action' in action:
                    action = 1
                elif 'fire-action' in  action:
                    action = 2
                elif 'right-action' in action:
                    action = 3
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward               
                state = observation
                if terminated or truncated:
                    break
            total_reward += episode_reward  
        average_reward = total_reward / self.num_episodes # [-500, 320]
        if np.isnan(average_reward) or np.isinf(average_reward):
            return 100.0
        else:
            sigmoid_val = 1 / (1 + np.exp(-average_reward / 100.0))
            normalized_fitness = 100.0 * (1.0 - sigmoid_val)
            return normalized_fitness

    def evaluate_test(self, ind):
        total_reward = 0
        seeds = [None] * self.num_episodes     
        for i in range(self.num_episodes):
            state, _ = self.env.reset(seed=seeds[i])
            episode_reward = 0
            for _ in range(self.max_steps):
                action = ind.LunarLander_interpret_mat(None, state, logExecutedNodes=True)    
                if not isinstance(action, str) or '-action' not in action:
                    action = 0
                    episode_reward += -105.0  # minimum reward in LunarLander
                    break 
                elif 'left-action' in action:
                    action = 1
                elif 'fire-action' in  action:
                    action = 2
                elif 'right-action' in action:
                    action = 3
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                state = observation
                if terminated or truncated:
                    break
            total_reward += episode_reward
        average_reward = total_reward / self.num_episodes
        if np.isnan(average_reward) or np.isinf(average_reward):
            return 100.0
        else:
            sigmoid_val = 1 / (1 + np.exp(-average_reward / 100.0))
            normalized_fitness = 100.0 * (1.0 - sigmoid_val)
            return normalized_fitness
class LunarLander1(LunarLander):
    def __init__(self, domain, npoints, ndimensions):
        super().__init__(domain, npoints, ndimensions)