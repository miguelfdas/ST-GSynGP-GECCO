import sys
import numpy as np
from copy import deepcopy
import GSynGP
from EA import *
from LCS_numba import *

class GP_Node:
    def __init__(self, value, params, sons, parent):
        self.value = value
        self.params = params
        self.sons = sons
        self.parent = parent
        self.height = 0  ##number of levels from this node to its furthest son
        if parent == None:
            self.depth = 0  ##number of levels from this node to the root
        else:
            self.depth = parent.depth + 1
        self.executed = False

    def toStr(self):
        s = self.value + '('
        for k in self.params.keys():
            s += k + '=' + self.params[k] + ','
        if s[-1] == ',':
            s = s[:-1]
        s += ')'
        return s

    def check_height(self):
        if(self.parent is None):
            self.depth = 0
        else:
            self.depth = self.parent.depth +1

        if self.sons is None:
            self.height = 0
        else:
            self.height = max([self.sons[i].height for i in range(len(self.sons))]) + 1
            
    def get_nodes(self, max_height, nodes):
        if self.height <= max_height:
            nodes.append(self)
        if self.sons is not None:
            for s in self.sons:
                s.get_nodes(max_height, nodes)
        return nodes

class GP_Individual(Individual):
    def __init__(self, genotype, phenotype, creation=None):
        Individual.__init__(self, genotype, phenotype, creation)
        self.current_node = self.phenotype
        self.current_node_arity = 0
        
        self.mean_genotypic_dist = 0
        self.mean_pgenotypic_dist = 0
        self.stddev_genotypic_dist = 0
        self.stddev_pgenotypic_dist = 0
        
        self.mean_behavioural_dist = 0
        self.stddev_behavioural_dist = 0
        self.mean_pbehavioural_dist = 0
        self.stddev_pbehavioural_dist = 0
        
        self.executedNodes = []        
        self.label = ''
        self.creation = -2

    def copy_indiv(self, original):
        self.genotype = deepcopy(original.genotype)
        self.phenotype = self.copy_node(original.phenotype, None)
        self.current_node = None
        self.fitness = None
        self.fitness_vals = []  # deepcopy(original.fitnesses_vals)
        self.creation = original.creation
        self.label = original.label
        self.size = original.size

    def copy_self(self):
        o = GP_Individual(None, None, None)
        o.copy_indiv(self)
        return o

    def copy_node(self, o_node, parent):
        n = GP_Node(deepcopy(o_node.value), deepcopy(o_node.params), None, parent)
        if o_node.sons is not None:
            n.sons = [self.copy_node(o_node.sons[i], n) for i in range(len(o_node.sons))]
            n.height = max([n.sons[i].height for i in range(len(n.sons))]) + 1
        return n  
        
    # 2025 Function to interpret a stateful tree with variable arities used in Santa Fe Trail and Los Altos Hills Ant Trail problem
    def stateful_interpret_ma(self, agent, current_node, logExecutedNodes=True):        
        if current_node is None:
            current_node = self.phenotype
        while True:    
            current_node.executed = True
            if logExecutedNodes and current_node not in self.executedNodes:
                self.executedNodes.append(current_node)                
            if current_node.sons is not None:
                if current_node.value == 'Progn':
                    current_node = current_node.sons[0]                       
                else:
                    if current_node.value == 'iffoodAhead':
                        val = agent.iffoodAhead()
                    else:
                        val = eval('agent.' + current_node.toStr())
                    if(val):
                        current_node = current_node.sons[0]
                    else:
                        current_node = current_node.sons[1]
            else:                
                motion_terminated = eval('agent.' + current_node.toStr())
                if motion_terminated:
                    motion_terminated = True
                    while current_node.parent is not None:
                        if current_node == current_node.parent.sons[0] and current_node.parent.value == 'Progn':
                            current_node = current_node.parent.sons[1]
                            break
                        else:
                            current_node = current_node.parent
                break
        return current_node, motion_terminated

    def arithmetic_interpret_mat(self, current_node, observation, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype
        current_node.executed = True
        if logExecutedNodes and current_node not in self.executedNodes:
            self.executedNodes.append(current_node)
        val = 0
        if current_node.sons is not None:
            if (len(current_node.sons) == 2): 
                if(current_node.value == 'add'):
                    val = self.arithmetic_interpret_mat(current_node.sons[0], observation, logExecutedNodes) + self.arithmetic_interpret_mat(current_node.sons[1], observation,logExecutedNodes)
                elif(current_node.value == 'sub'):
                    val = self.arithmetic_interpret_mat(current_node.sons[0], observation,logExecutedNodes) - self.arithmetic_interpret_mat(current_node.sons[1], observation,logExecutedNodes)
                elif(current_node.value == 'mult'):
                    val = self.arithmetic_interpret_mat(current_node.sons[0], observation,logExecutedNodes) * self.arithmetic_interpret_mat(current_node.sons[1], observation,logExecutedNodes)
                elif(current_node.value == 'div'):
                    den = self.arithmetic_interpret_mat(current_node.sons[1], observation,logExecutedNodes)
                    if(den ==0):
                        val = 1
                    else:
                        val = self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes)*1.0/ den
            elif(len(current_node.sons) == 1):  
                if(current_node.value == 'sin'):
                    try:
                        val = np.sin(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes))
                    except:
                        val = 1
                elif(current_node.value == 'cos'):
                    try:
                        val = np.cos(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes))
                    except:
                        val = 1
                elif(current_node.value == 'exp'):
                    try:
                        val = np.exp(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes))
                    except:
                        val = 0
                elif(current_node.value == 'lnmod'):
                    try:
                        val = abs(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes))
                        if(val !=0):
                            val = np.log(val)
                    except:
                        val = 0
                elif(current_node.value == 'sqrt'):
                    try:
                        val = np.sqrt(abs(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes)))
                    except:
                        val = 1
                elif(current_node.value=='plog'):
                    try:
                        val = abs(self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes))
                        if(val !=0):
                            val = np.log10(val)
                    except:
                        val = 0
                elif(current_node.value == 'ppow'):
                    try:
                        x1 = self.arithmetic_interpret_mat(current_node.sons[0],observation, logExecutedNodes)
                        x2 = self.arithmetic_interpret_mat(current_node.sons[1],observation, logExecutedNodes)
                        if(x1!=0 or (x1==x2 and x1==0)):
                            val = np.power(abs(x1), x2)
                        else:
                            val = 0
                    except:
                        val = 0
                if np.isinf(val):
                    if(val >0):
                        val=sys.maxsize
                    else:
                        val = -sys.maxsize
        else:
            if(isinstance(current_node.value, str)):
                if('constant' in current_node.value):
                    val = current_node.params['value']
                else:
                    i = current_node.value.split('_')[1]
                    if(i=='all'):
                        val= np.zeros(observation.shape) + observation
                    else:
                        val = observation[int(i)]
            else:
                val = current_node.value

        return val
    
    # 2025 Function to interpret a boolean tree used to N-Multiplexer benchmark problems
    # Os interpertadores tem que ser feitos manualmente e adaptados ao numero de filhos de cada simbolo é isso que queremos?
    def boolean_interpret_mat(self, current_node, observation, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype
        current_node.executed = True
        if logExecutedNodes and current_node not in self.executedNodes:
            self.executedNodes.append(current_node)
        val = 0
        if current_node.sons is not None:
            if current_node.value == 'if-then-else-boolean':
                if self.boolean_interpret_mat(current_node.sons[0], observation, logExecutedNodes):
                    val = self.boolean_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                else:
                    val = self.boolean_interpret_mat(current_node.sons[2], observation, logExecutedNodes)
            elif current_node.value == 'and':
                val = self.boolean_interpret_mat(current_node.sons[0], observation, logExecutedNodes) and self.boolean_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
            elif current_node.value == 'or':
                val = self.boolean_interpret_mat(current_node.sons[0], observation, logExecutedNodes) or self.boolean_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
            elif current_node.value == 'not':
                val = not self.boolean_interpret_mat(current_node.sons[0], observation, logExecutedNodes)
        else:
            if "bit_" in current_node.value:    
                i = current_node.value.split('_')[1]
                val = observation[int(i)]
        return val
    
    # 2025 Function to interpret a tree used to Spambase benchmark problem
    def spambase_interpret_mat(self, current_node, observation, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype
        current_node.executed = True
        if logExecutedNodes and current_node not in self.executedNodes:
            self.executedNodes.append(current_node)
        val = None
        aux = None
        aux1 = None
        aux2 = None
        if current_node.sons is not None:
            if (len(current_node.sons) == 3):
                if(current_node.value == 'if-then-else'):
                    aux = bool(self.spambase_interpret_mat(current_node.sons[0], observation,logExecutedNodes))
                    if(aux):
                        val = float(self.spambase_interpret_mat(current_node.sons[1], observation,logExecutedNodes))
                    else:
                        val = float(self.spambase_interpret_mat(current_node.sons[2], observation,logExecutedNodes))      
            elif (len(current_node.sons) == 2):
                if(current_node.value == 'add'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = float(aux1 + aux2)
                elif(current_node.value == 'sub'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = float(aux1 - aux2)
                elif(current_node.value == 'mult'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = float(aux1 * aux2)
                elif(current_node.value == 'div'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    if aux2 != 0:
                        val = float(aux1 / aux2)
                    else:
                        val = float(1)
                elif(current_node.value == 'and'):
                    aux1 = bool(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = bool(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = bool(aux1 and aux2)
                elif(current_node.value == 'or'):
                    aux1 = bool(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = bool(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = bool(aux1 or aux2)
                elif(current_node.value == 'lt'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = bool(aux1 < aux2)
                elif(current_node.value == 'eq'):
                    aux1 = float(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    aux2 = float(self.spambase_interpret_mat(current_node.sons[1], observation, logExecutedNodes))
                    val = bool(aux1 == aux2)
            elif (len(current_node.sons) == 1):
                if(current_node.value == 'not'):
                    aux = bool(self.spambase_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                    val = bool(not aux)
        else:                       
            if(isinstance(current_node.value, str)):
                if('constant' in current_node.value):
                    val = float(current_node.params['value'])
                elif('bit' in current_node.value):
                    val = bool(current_node.params['value'])
                elif('var' in current_node.value):
                    i = current_node.value.split('_')[1]
                    aux = observation[int(i)]
                    val = float(aux)
        return val

    def LunarLander_interpret_mat(self, current_node, observation, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype
        current_node.executed = True
        if logExecutedNodes and current_node not in self.executedNodes:
            self.executedNodes.append(current_node)
        val = None
        aux1 = None
        aux2 = None
        
        if current_node.sons is not None:
            if (len(current_node.sons) == 3):
                if (current_node.value == 'if-then-else-action'):
                    if(self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes)):
                        val = self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                    else:
                        val = self.LunarLander_interpret_mat(current_node.sons[2], observation, logExecutedNodes)
                elif(current_node.value == 'if-then-else'):
                    if(self.LunarLander_interpret_mat(current_node.sons[0], observation,logExecutedNodes)):
                        val = self.LunarLander_interpret_mat(current_node.sons[1], observation,logExecutedNodes)
                    else:
                        val = self.LunarLander_interpret_mat(current_node.sons[2], observation,logExecutedNodes)
            elif (len(current_node.sons) == 2):
                if(current_node.value == 'add'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) + self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'sub'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) - self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'mult'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) * self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'div'):
                    aux1 = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes)
                    aux2 = self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                    if aux2 != 0:
                        val = aux1 / aux2
                    else:
                        val = 1
                elif(current_node.value == 'and'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) and self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'or'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) or self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'gt'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) > self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'lt'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) < self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
                elif(current_node.value == 'eq'):
                    val = self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes) == self.LunarLander_interpret_mat(current_node.sons[1], observation, logExecutedNodes)
            elif (len(current_node.sons) == 1):
                if(current_node.value == 'not'):
                    val = not self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes)
                elif(current_node.value == 'cos'):
                    val = np.cos(self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                elif(current_node.value == 'sin'):
                    val = np.sin(self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
                elif(current_node.value == 'tan'):
                    val = np.tan(self.LunarLander_interpret_mat(current_node.sons[0], observation, logExecutedNodes))
        else:                       
                if('x-cord' in current_node.value):
                    val = observation[0]
                elif('y-cord' in current_node.value):
                    val = observation[1]
                elif('x-velocity' in current_node.value):
                    val = observation[2]
                elif('y-velocity' in current_node.value):
                    val = observation[3]
                elif('angle' in current_node.value):
                    val = observation[4]
                elif('angular-velocity' in current_node.value):
                    val = observation[5]
                elif('left-leg-contact' in current_node.value):
                    val = observation[6]
                elif('right-leg-contact' in current_node.value):
                    val = observation[7]
                elif('constant' in current_node.value):
                    val = current_node.params['value']
                elif('nothing' in current_node.value):
                    val = 'nothing'
                elif('fire-action' in current_node.value):
                    val = 'fire-action'
                elif('left-action' in current_node.value):
                    val = 'left-action'
                elif('right-action' in current_node.value):
                    val = 'right-action'
        return val  

    def toList(self, l, current_node):
        s = current_node.value + '('
        keys = list(current_node.params.keys())
        keys.sort()
        for i in range(len(keys)):
            p = keys[i]
            if i < len(keys) - 1:
                s += p + '=' + str(current_node.params[p]) + ','
            else:
                s += p + '=' + str(current_node.params[p])
        s += ')'
        l.append(s)
        if current_node.sons is not None:
            for i in range(len(current_node.sons)):
                l = self.toList(l, current_node.sons[i])
        return l

    def toTree(self, genotype, curr_ind, parent, function_set, function_mat): 
        if curr_ind >= len(genotype):
            return [None, curr_ind]
        
        val = genotype[curr_ind]
                
        curr_ind += 1 
        value = val[:val.index('(')]
        
        if ':' in val:
            val = val.split(':')[0]
        
        if '_' in val:
            params = val[val.index('(') + 1:].split('_')[0]
        else:
            params = val[val.index('(') + 1:]
                    
        val = [value, params]
        n = GP_Node(val[0], {}, None, parent)

        if val[1] != ')':
            val[1] = val[1][:-1].split(',')
            for v in val[1]:
                v = [v[:v.index('=')], v[v.index('=') + 1:]]
                try:
                    n.params[v[0]] = int(v[1])
                except:
                    try:
                        n.params[v[0]] = float(v[1])
                    except:
                        n.params[v[0]] = v[1]

        if val[0] in function_set:
            n.sons = []
            arity = function_mat[val[0]]['arity']
            for _ in range(arity):
                if curr_ind < len(genotype):
                    [s, curr_ind] = self.toTree(genotype, curr_ind, n, function_set, function_mat)
                    n.sons.append(s)

        n.check_height()
        return [n, curr_ind]

    def get_terminals(self, l, current_node):
        if current_node.sons is None:
            l.append(current_node)
        else:
            for s in current_node.sons:
                self.get_terminals(l, s)

    def get_functions(self, l, current_node):
        if current_node.sons is not None:
            l.append(current_node)
            for s in current_node.sons:
                if s.sons is not None:
                    self.get_functions(l, s)

    def make_genotype(self):
        self.genotype = []
        self.genotype = self.toList(self.genotype, self.phenotype)
        self.genotype = np.array(self.genotype, dtype='<U100')

class GP(EA):
    def __init__(self, pop_size, generations, ei, ri, p_cross, p_mut, tourn_size, elite_size, evaluation_function, p_reeval, log_file, max_depth, terminal_set,
                 function_set, function_mat, terminal_mat, terminal_params, data_types, iteration, mutation_sigma, params_coefs=(1,1), maximise=False, compute_diversities=False, n_instances = 1, prob_type=None):
        EA.__init__(self, pop_size, generations, ei, ri, p_cross, p_mut, tourn_size, elite_size, evaluation_function, p_reeval, log_file, genotypic_distance, compute_diversities,compute_diversities,params_coefs, maximise, n_instances, prob_type)

        self.max_depth = max_depth
        self.terminal_set = terminal_set
        self.function_set = np.array(function_set)
        self.function_mat = function_mat
        self.terminal_mat = terminal_mat
        self.terminal_params = terminal_params
        self.data_types = data_types
        self.iteration = iteration
        self.mean_executed_nodes = 0
        self.std_executed_nodes = 0
        self.mean_percentage_executed_nodes = 0
        self.std_percentage_executed_nodes = 0
        self.mutation_sigma = mutation_sigma

    def log(self, gen):
        if (gen <= 0):
            f = open(self.log_file, 'w')
            
            s = 'generation\tmean_pop_size\tstd_pop_size\tmean_size_diff\tstd_size_diff\tmean_genotypic_dist\tstd_genotypic_dist\tmean_pgenotypic_dist\tstd_pgenotypic_dist'
            s += '\tmean_behavioural_dist\tstd_behavioural_dist\tmean_pbehavioural_dist\tstd_pbehavioural_dist'
            s += '\tmean_pop_fitness\tstd_pop_fitness\tmean_pop_fitness_test\tstd_pop_fitness_test\tbest_fitness_test'
            s += '\tmean_executed_nodes\tstd_executed_nodes\tmean_pexecuted_nodes\tstd_pexecuted_nodes\tmean_gsyngp_iterations\tstd_gsyngp_iterations'
            s += "\tmean_n_functions\tstd_n_functions\tmean_n_non_arith\tstd_n_non_arith\tmean_max_seq\tstd_max_seq\tmean_depth\tstd_depth"
            
            headers = self.best_indiv.get_headers()
            for h in headers:
                s += '\t' + 'bestIndiv_' + h
            f.write(s + '\n')
        else:
            f = open(self.log_file, 'a')
        try:
            ind_iters = int(self.ind_iterations[gen])
            mean_iterations = np.mean(self.gsyngp_iterations[gen][:ind_iters])
            std_iterations = np.std(self.gsyngp_iterations[gen][:ind_iters])            
            ind_sd = int(self.ind_size_diff[gen])
            mean_size_diff = np.mean(self.size_diff[gen][:ind_sd])
            std_size_diff = np.std(self.size_diff[gen][:ind_sd])
        except Exception as e:
            mean_iterations = 0
            std_iterations = 0            
            mean_size_diff = 0
            std_size_diff = 0

        s = str(gen) + '\t' + str(self.mean_pop_size) + '\t' + str(self.std_pop_size) + '\t' + str(mean_size_diff) + '\t' + str(std_size_diff)
        s += '\t' + str(self.mean_genotypic_dist) + '\t' + str(self.stddev_genotypic_dist) + '\t' + str(self.mean_pgenotypic_dist) + '\t' + str(self.stddev_pgenotypic_dist)
        s += '\t' + str(self.mean_behavioural_diversity) + '\t' + str(self.std_behavioural_diversity) + '\t' + str(self.mean_pbehavioural_diversity) + '\t' + str(self.std_pbehavioural_diversity) + '\t' + str(self.mean_fitness) + '\t' + str(self.std_fitness)
        s += '\t' + str(self.mean_fitness_test) + '\t' + str(self.std_fitness_test) + '\t' + str(self.best_fitness_test)
        s += '\t' + str(self.mean_executed_nodes) + '\t' + str(self.std_executed_nodes) + '\t' + str(self.mean_percentage_executed_nodes) + '\t' + str(self.std_percentage_executed_nodes) + '\t' +str(mean_iterations)+ '\t' +str(std_iterations)
        s += '\t' + str(self.mean_n_functions) + '\t' + str(self.std_n_functions) + '\t' + str(self.mean_n_non_arith) + '\t' + str(self.std_n_non_arith) 
        s += '\t' + str(self.mean_max_seq) + '\t' + str(self.std_max_seq) + '\t' + str(self.mean_depth) + '\t' + str(self.std_depth)        
        s += '\t' + self.best_indiv.toStringCompleteSingleLine()

        f.write(s + '\n')
        f.close()

    # 2025 Alterated to use crossover3_ma -> Function to perform crossover with variable arities
    def GSynGP_crossover(self, indA, indB, curr_offsprings, max_offsprings, gen):
        result = GSynGP.crossover3_mat(indA.genotype, indB.genotype, self.function_mat, self.terminal_mat, self.iteration, curr_offsprings, max_offsprings, gen)
        a = result[0]
        iterations_performed = result[1]
        size_diff = result[2]
        a = GP_Individual(a, None, None)
        [phenotype, curr_ind] = a.toTree(a.genotype, 0, None, self.function_set, self.function_mat)
        a.phenotype = phenotype
        return [a],[iterations_performed],[size_diff]

    def crossover(self, indA, indB, curr_offsprings, max_offsprings, gen):
        indA = indA.copy_self()
        indB = indB.copy_self()        

        funcs, terms = [], []
        indA.get_functions(funcs, indA.phenotype)
        indA.get_terminals(terms, indA.phenotype)
        
        if random.random() < 0.9:
            subA = random.choice(funcs)
            expected_type = self.function_mat[subA.value]['return']
        else:
            subA = random.choice(terms)
            expected_type = self.terminal_type_mat(subA, self.terminal_mat)
                        
        funcs, terms = [], []
        indB.get_functions(funcs, indB.phenotype)
        indB.get_terminals(terms, indB.phenotype)
        
        if random.random() < 0.9:
            cands = []
            for f in funcs:
                ret_type = self.function_mat[f.value]['return'] 
                if ret_type == expected_type and f.height <= self.max_depth - subA.depth:
                    cands.append(f)
        else:
            cands = []
            for i in terms:
                if self.terminal_type_mat(i, self.terminal_mat) == expected_type:
                    cands.append(i)    

        if len(cands) == 0:
            return [indA], [0], [0]
        else:
            subB = random.choice(cands)
            
        subBnodes = []
        subB.get_nodes(subB.height, subBnodes)
        subAnodes = []
        subA.get_nodes(subA.height, subAnodes)
        
        if subA.parent is None:
            return [indB],[0],[0]
        for i in range(len(subA.parent.sons)):
            if subA is subA.parent.sons[i]:
                subA.parent.sons[i] = subB

        subB.parent = subA.parent
        subB.depth = subA.depth
        p = subB.parent
        while p is not None:
            p.check_height()
            p = p.parent

        return [indA], [0], [len(subBnodes) - len(subAnodes)]
    
    # 2025 Function to get the type of a terminal symbol, used in the ramped half-and-half initialization method with variable arities  
    def terminal_type_symbol_mat(self, symbol, terminal_mat):
        for i in terminal_mat.keys():
            if i in symbol:
                return terminal_mat[i]['type']

    # 2025 Function to get the type of a terminal, it's a not the best way but its working for now
    def terminal_type_mat(self, node, terminal_mat):
        for i in terminal_mat.keys():
            if i in node.value:
                return terminal_mat[i]['type']

    # 2025 Alterated version of the ramped half-and-half initialization method to use random_individual_mat instead of random_individual to generate new individuals with varaible arities and types
    def ramped_half_and_half_mat(self, max_depth):
        def pop_initialization(pop_size):
            population = []
            depths = [i + 1 for i in range(max_depth - 1)]
            depths.reverse()
            l = len(depths)
            for i in range(pop_size):
                expected_type = random.choice(self.data_types)
                population.append(GP_Individual(None, self.random_individual_mat(0, depths[i % l], None, random.random() < 0.5, expected_type), {}))
                population[-1].make_genotype()
            return population
        return pop_initialization    
    
    #2025 Function to generate a random individual with variable arities and types
    def random_individual_mat(self, current_depth, max_depth, parent, grow, expected_type):
        n = GP_Node(None, {}, None, parent)        
            
        if current_depth == max_depth or (grow and random.random() < (current_depth * 1.0) / max_depth):
            valid_terminals = [t for t in self.terminal_set if expected_type in self.terminal_type_symbol_mat(t, self.terminal_mat)]
            n.value = random.choice(valid_terminals)
            n.sons = None      
            if n.value in self.terminal_params.keys():
                for p in self.terminal_params[n.value].keys():
                    n.params[p] = random.choice(self.terminal_params[n.value][p]) 
        else:
            valid_functions = [f for f in self.function_set if expected_type in self.function_mat[f]['return']] 
            n.value = random.choice(valid_functions)      
            arg_types = self.function_mat[n.value]['args']
            n.sons = [self.random_individual_mat(current_depth + 1, max_depth, n, grow, arg_type) for arg_type in arg_types]
            n.height = max(son.height for son in n.sons) + 1
        return n

    def mutate_node_mat(self, n, is_terminal):
        if is_terminal:
            if len(self.terminal_set) > 1:
                if random.random() > 0.5:
                    mutated = self.change_node_params(n, self.terminal_params)
                else:   
                    candidates = []
                    for i in self.terminal_set:
                        if self.terminal_type_symbol_mat(i, self.terminal_mat) == self.terminal_type_mat(n, self.terminal_mat):
                            candidates.append(i)
                    mutated = self.change_node_symbol_mat(n, candidates, self.terminal_params)
            else:
                mutated = self.change_node_params(n, self.terminal_params)
        else:
            candidates = []
            for i in self.function_set:
                if self.function_mat[i]['return'] == self.function_mat[n.value]['return']:
                    if self.function_mat[i]['args'] == self.function_mat[n.value]['args']:
                        candidates.append(i)
            function_params = {}
            mutated = self.change_node_symbol_mat(n, candidates, function_params)
        return n, mutated

    def uniform_node_mutation(self, ind):
        nodes=[]
        ind.get_terminals(nodes, ind.phenotype)    
        ind.get_functions(nodes, ind.phenotype)
        mutated = False
        while (len(nodes) > 0 and not mutated):
            n = random.choice(nodes)
            nodes.remove(n)
            is_terminal = (n.value in self.terminal_set)
            n, mutated = self.mutate_node_mat(n, is_terminal)
        return ind


    def biased_node_mutation(self, ind):
        terminals, functions=[],[]
        ind.get_functions(functions, ind.phenotype)
        ind.get_terminals(terminals, ind.phenotype)    
        mutated = False
        while ((len(terminals) > 0 or len(functions) > 0) and not mutated):
            if(len(functions) > 0 and random.random()<0.9):
                nodes = functions
                is_terminal = False
            else:
                nodes = terminals
                is_terminal = True
            n = random.choice(nodes)
            nodes.remove(n)
            n, mutated = self.mutate_node_mat(n, is_terminal)
        return ind

    def change_node_params(self, node, params):
        mutated = False
        if (len(node.params.keys()) == 0):
            return mutated
        p_cands = list(node.params.keys())
        random.shuffle(p_cands)
        i=0
        while(not mutated and i<len(p_cands)): 
            p = p_cands[i]
            s = node.params[p]
            if isinstance(s, str):
                try:
                    s = eval(s)
                    vals = [eval(k) for k in params[node.value][p]]
                    high = max(vals)
                    low = min(vals)
                    domain_width = high-low
                    if isinstance(s, float):
                        node.params[p] = str(min(high, max(low, round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width), 3))))
                    elif isinstance(s, int):
                        node.params[p] = str(min(high, max(low, int(round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width))))))
                    mutated = True
                    break
                except Exception as e:
                    mutated = False
            else:                
                high = max(params[node.value][p])
                low = min(params[node.value][p])
                domain_width = high-low
                if isinstance(s, float):
                    node.params[p] = min(high, max(low, round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width), 3)))
                    mutated = True
                    break
                elif isinstance(s, int):
                    node.params[p] = min(high, max(low, int(round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width)))))
                    mutated = True
                    break
            if not mutated:
                v_cands = list(params[node.value][p])
                random.shuffle(v_cands)
                for v in v_cands:
                    if (v != node.params[p]):
                        node.params[p] = v
                        mutated = True
                        break
            i+=1
        return mutated
    
    # 2025 Function to change the symbol of a node with variable arities
    def change_node_symbol_mat(self, node, symbols, params):
        if len(symbols) == 1:
            return False
        
        mutated = False
        s_cands = symbols
        random.shuffle(s_cands)    
        for v in s_cands:
            if v != node.value:
                node.value = v
                nkeys = list(node.params.keys())
                if v not in params.keys():
                    for p in nkeys:
                        del node.params[p]
                else:
                    for p in nkeys:
                        if p not in params[v].keys():
                            del node.params[p]
                    for p in params[v].keys():
                        if p not in node.params.keys() or node.params[p] not in params[v][p]:
                            node.params[p] = random.choice(params[v][p])
                    mutated = True
                    break
        return mutated
    
#----------------------------------------------------------------------------------------------------------------------------------

def genotypic_distance(A, B, float_vals=False):
    C = LCS(A, B)

    [ma, mb, na, nb] = LCS_MASKS(A, B, C)
    return na, nb