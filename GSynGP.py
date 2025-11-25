import numpy as np
from copy import deepcopy
import random
from LCS_numba import *
from GP import *

class node:
    def __init__(self, v, s,depth):
        self.value = v
        self.sons = s
        self.depth=depth

def count_notations(ma):
    af = 0
    at = 0
    for c in ma:
        if 'F_' in c:
            af += 1
        elif 'T_' in c:
            at += 1
    return af, at

def compute_distance(ma, mb):
    af, at = count_notations(ma)
    bf, bt = count_notations(mb)

    ift = min(bf, bt)
    aft = min(af, at)
    sf = af - aft
    st = at - aft

    d = ift + aft + sf + st
    return d

def isnumber(x):
    try:
        a = float(x)
        return True
    except:
        return False

# 2025 Function to check the individual     
def checker_mat(function_mat, terminal_mat, ind, index, arg):    
    if index == len(ind):
        return None, 0
    
    val = ind[index]
    if '(' in val:
        val = val.split('(')[0]

    if arg == None:
        flag = 1
    else:
        flag = 0

    if val in function_mat:
        if flag == 1 or function_mat[val]['return'] == arg: 
            args = function_mat[val]['args']
            sons = []
            v = ind[index]
            for arg in args:
                child, index = checker_mat(function_mat, terminal_mat, ind, index + 1, arg)
                if child is None:
                    return None, 0
                sons.append(child)          
            d = max(child.depth for child in sons) + 1
            n = node(v, sons, d)
            return n, index
        else:
            return None, 0
    else:
        if '_' in val:
            val = val.split('_')[0]
        if flag == 1 or terminal_mat[val]['type'] == arg:
            if isnumber(ind[index]):
                return node(float(ind[index]), None, 0), index
            else:
                return node(ind[index], None, 0), index 
        else:
            return None, 0

# 2025 Function to print the individual with variable arities
def print_tree_preorder_mat(root, s):
    if isnumber(root.value):
        s.append(str(root.value))
    else:
        s.append(root.value)
    if root.sons is not None:
        for child in root.sons:
            print_tree_preorder_mat(child, s)
    return s

# 2025 Function to check the individual
def check_indiv_mat(ind, function_mat, terminal_mat):
    t, i = checker_mat(function_mat, terminal_mat, ind, 0, None)
    
    if t != None:
        s = np.array(print_tree_preorder_mat(t, []))
        if not different(ind, s):
            return True, None
        else:
            return True, s
    return False, None

def concat(exA):
    s = []
    for i in range(len(exA)):            
        if exA[i].strip() != '':
            if 'F_' in exA[i] or 'T_' in exA[i]:
                s.append(exA[i][exA[i].index('_') + 1:])
            else:
                s.append(exA[i].strip())

    return np.array(s, dtype=exA.dtype)

# 2025 Function to check the type of the node
def clean_mat(s):
    if ':' in s:
        s = s.split(':')[0]
    if '(' in s:
        s = s.split('(')[0]
    if 'T_' in s:
        s = s.split('_')[1]       
    if 'F_' in s:
        s = s.split('_')[1]
    if 'var' in s:
        s = s.split('_')[0]
    if '_' in s:
        s = s.split('_')[0]
    return s

def different(A,B):
    if(len(A)!=len(B)):
        return True
    for i in range(len(A)):
        if(A[i]!=B[i]):
            return True
    return False

def compute_masks(A, B, function_mat, terminal_mat):
    rA, rB = toStructure(A, list(function_mat.keys())), toStructure(B, list(function_mat.keys()))
    C = LCS(A, B)
    [ma, mb, na, nb] = LCS_MASKS(A, B, C)
    [exA, exB] = expanded_masks(ma, mb, A, B, rA,rB, function_mat, terminal_mat)
    return rA, rB, C, ma,mb, exA, exB

def existances(exM):
    T_, F_ = False, False
    for s in exM:
        if len(s)>2:
            if(s[:2] == 'T_'):
                T_ = True
            elif(s[:2] == 'F_'):
                F_ = True
    return F_, T_

def replaceT(exA, exB, function_mat, terminal_mat):
    indA = []
    indB = []
    for i in range(len(exA)):
        if 'T_' in exA[i]:
            indA.append(i)
        if 'T_' in exB[i]:
            indB.append(i)
    random.shuffle(indB)
    for iB in indB:
        oldIBA = deepcopy(exA)
        oldIBB = deepcopy(exB)
        exB[iB] = exB[iB][2:]
        if exA[iB] == '   ':
            random.shuffle(indA)
            for iT in indA:    
                if terminal_mat[clean_mat(exA[iT])]['type'] == terminal_mat[clean_mat(exB[iB])]['type']:
                    exA[iB] = exB[iB]
                    exA[iT] = '   '
                    ind = concat(exA)
                    flag, s = check_indiv_mat(ind, function_mat, terminal_mat)
                    if flag:
                        if s is None:
                            return ind, False
                        else:
                            return s, True
                    else:
                        exA = oldIBA
                        exB = oldIBB
        elif 'T_' in exA[iB]:
            if terminal_mat[clean_mat(exA[iB])]['type'] == terminal_mat[clean_mat(exB[iB])]['type']:
                exA[iB] = exB[iB]
                return concat(exA), False
        else:
            random.shuffle(indA)
            for iT in indA:
                if terminal_mat[clean_mat(exA[iT])]['type'] == terminal_mat[clean_mat(exB[iB])]['type']:
                    exA[iB] = exB[iB]
                    return concat(exA), False
    return None, False

def deleteFT(exA, exB, function_mat, terminal_mat):
    indF = []
    indT = []
    for i in range(len(exA)):
        if 'F_' in exA[i]:
            indF.append(i)
        elif 'T_' in exA[i]:
            indT.append(i)   
    random.shuffle(indF)
    for iF in indF:  
        oldexA = deepcopy(exA)
        oldexB = deepcopy(exB)
        arity = function_mat.get(exA[iF].split('(')[0].split('_')[1], 0).get('arity', 0)
        args = function_mat.get(exA[iF].split('(')[0].split('_')[1], 0).get('args', 0)           
        exA[iF] = '   '
        exB[iF] = exB[iF][2:]
        random.shuffle(indT)
        spaces_inserted = 0        
        for iT in indT:
                if iT > iF and spaces_inserted < arity - 1 and terminal_mat[clean_mat(exA[iT])]['type'] == args[spaces_inserted]:
                    exA[iT] = '   '
                    exB[iT] = exB[iT][2:]
                    spaces_inserted += 1                
        ind = concat(exA)
        flag, s = check_indiv_mat(ind, function_mat, terminal_mat)
        if flag:
            if s is None:
                return ind, False
            else:
                return s, True
        else:
            exA = oldexA
            exB = oldexB
    return None, False

def insertFT(exA, exB, function_mat, terminal_mat):
    indBF = []
    indBT = []
    for i in range(len(exA)):
        if 'F_' in exB[i] and '   ' in exA[i]:
            indBF.append(i)
        elif 'T_' in exB[i] and '   ' in exA[i]:
            indBT.append(i)   
    random.shuffle(indBF)
    for iF in indBF:
        if exA[iF] == '   ':            
            oldexA = deepcopy(exA)
            oldexB = deepcopy(exB)
            exA[iF] = exB[iF][exB[iF].index('_') + 1:]
            exB[iF] = exB[iF][2:]          
            arity = function_mat.get(exA[iF].split('(')[0], 0).get('arity', 0)
            args = function_mat.get(exA[iF].split('(')[0], 0).get('args', 0)             
            random.shuffle(indBT)
            terminals_inserted = 0
            for iT in indBT:
                if iT > iF and exA[iT] == '   ' and terminals_inserted < arity - 1 and terminal_mat[clean_mat(exB[iT])]['type'] == args[terminals_inserted]: 
                    exA[iT] = exB[iT][exB[iT].index('_') + 1:]
                    exB[iT] = exB[iT][2:]
                    terminals_inserted += 1     
            ind = concat(exA)
            flag, s = check_indiv_mat(ind, function_mat, terminal_mat)
            if flag:
                if s is None:
                    return ind, False
                else:
                    return s, True
            else:
                exA = oldexA
                exB = oldexB
    return None, False

def replaceF(exA, exB, function_mat, terminal_mat):
    indA = []
    indB = []
    for i in range(len(exA)):
        if 'F_' in exA[i]:
            indA.append(i)
        if 'F_' in exB[i]:
            indB.append(i)
    random.shuffle(indB)
    for iB in indB:
        oldIBA = deepcopy(exA)
        oldIBB = deepcopy(exB)
        exB[iB] = exB[iB][2:]
        if '   ' in exA[iB]:
            random.shuffle(indA)
            for iF in indA:
                if function_mat[clean_mat(exA[iF])]['return'] == function_mat[clean_mat(exB[iB])]['return']:
                    if function_mat[clean_mat(exA[iF])]['args'] == function_mat[clean_mat(exB[iB])]['args']:
                        exA[iB] = exB[iB]
                        exA[iF] = '   '
                        ind = concat(exA)
                        flag, s = check_indiv_mat(ind, function_mat, terminal_mat)
                        if flag:
                            if s is None:
                                return ind, False
                            else:
                                return s, True
                        else:
                            exA = oldIBA
                            exB = oldIBB
        elif 'F_' in exA[iB]:
            if function_mat[clean_mat(exA[iB])]['return'] == function_mat[clean_mat(exB[iB])]['return']:
                if function_mat[clean_mat(exA[iB])]['args'] == function_mat[clean_mat(exB[iB])]['args']:
                    exA[iB] = exB[iB]
                    return concat(exA), False
        else:
            random.shuffle(indA)
            for iF in indA:
                if function_mat[clean_mat(exA[iF])]['return'] == function_mat[clean_mat(exB[iB])]['return']:
                    if function_mat[clean_mat(exA[iF])]['args'] == function_mat[clean_mat(exB[iB])]['args']:
                        exA[iF] = exB[iB]
                        return concat(exA), False
    return None, False

# 2025 Function to perform crossover with variable arities use the functions with _ma suffix
def crossover3_mat(A, B, function_mat, terminal_mat, iteration, curr_offsprings, max_offsprings, gen):        
    A = deepcopy(A)
    B = deepcopy(B)
    
    it = 0
    orig_size = len(A)
        
    if(not different(A, B)):
        return A, 0, 0

    rA, rB, C, ma, mb, exA, exB = compute_masks(A, B, function_mat, terminal_mat)

    if iteration == 'random':
        times = compute_distance(exA, exB) * 0.5 * random.random() 
    elif iteration == 'half':
        times = compute_distance(exA, exB) * 0.5 
    else:
        times = eval(iteration)

    while it < times and different(A,B):
        options = []

        aF, aT = existances(exA)
        bF, bT = existances(exB)

        if(aF and aT):
            options.append(deleteFT)
        if(bF and bT):
            options.append(insertFT)
        if options == []:
            if(aT and bT):
                options.append(replaceT)
            if(bF and aF):
                options.append(replaceF)
        
        random.shuffle(options)

        for choice in options:
            off, flag = choice(exA, exB, function_mat, terminal_mat)
            
            if flag:
                rA, rB, C, ma, mb, exA, exB = compute_masks(A, B, function_mat, terminal_mat)
            if(off is not None):
                A = off
                break
            
        it += 1
        
    return A, it, len(A) - orig_size