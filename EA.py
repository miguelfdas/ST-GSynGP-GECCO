import random
from multiprocessing import Process, Queue
import multiprocessing
from EvaluationFunctions import *
import time
import os
import numpy as np

import gymnasium as gym

class Individual:
    def __init__(self, g, p, creation=None):
        self.genotype = g
        self.phenotype = p
        self.fitness = None
        self.size = 0
        self.fitness_vals=[]
        self.behaviour = np.zeros(0)
        self.creation = creation

        self.genotypic_distances = None
        self.pgenotypic_distances = None
        self.mean_genotypic_dist = -1
        self.stddev_genotypic_dist = -1
        self.mean_pgenotypic_dist = -1
        self.stddev_pgenotypic_dist = -1
        
        self.behavioural_distances = None
        self.pbehavioural_distances = None
        self.mean_behavioural_dist = -1
        self.stddev_behavioural_dist = -1
        self.mean_pbehavioural_dist = -1
        self.stddev_pbehavioural_dist = -1

    def toString(self):
        return str(self.genotype)

    def toStringComplete(self):
        s = '<ind>\n'

        ###list all attributes of a class
        fields = self.__dict__.keys()
        for f in fields:
            if( f!= 'headers'):
                try:
                    s+= '<' + f + '>' + str(self.__dict__[f].tolist()) + '</' + f + '>\n'
                except:
                    s+= '<' + f + '>' + str(self.__dict__[f]) + '</' + f + '>\n'

        s += '</ind>'
        return s

    def get_headers(self):
        self.headers = list(self.__dict__.keys())
        self.headers.sort()
        if('headers' in self.headers):
            self.headers.remove('headers')
        return self.headers

    def toStringCompleteSingleLine(self):
        self.get_headers()
        s = ''
        for f in self.headers:
            if( f!= 'headers'):
                try:
                    s+= str(self.__dict__[f].tolist()) + '\t'
                except:
                    s+= str(self.__dict__[f]) + '\t'
        return s[:-1]


    def add_fitness(self, val):
        try:
            self.fitness_vals.append(val)
        except:
            self.fitness_vals = self.fitness_vals.tolist()
            self.fitness_vals.append(val)
        self.fitness = np.mean(self.fitness_vals)

class EA:
    def __init__(self, pop_size, generations, ei, ri, p_cross, p_mut, tourn_size, elite_size, evaluation_function, p_reeval, log_file, genotypic_distance, compute_gp_metrics=False, compute_diversities=False, params_coefs=(1,1), maximise=False, instances=1, prob_type='Boolean'):
        multiprocessing.set_start_method('fork', force=True)
      
        self.genotypic_distance = genotypic_distance
        self.compute_gp_metrics = compute_gp_metrics
        self.compute_diversities = compute_diversities
        self.tourn_size = tourn_size
        self.maximise=maximise
        self.pop_size = pop_size
        self.generations = generations
        self.ei = ei
        self.ri = ri
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.elite_size = elite_size

        self.p_reeval = p_reeval

        self.n_offsprings = self.pop_size - self.ei - self.ri
        self.log_file = log_file
        self.population = None
        self.best_indiv = None
        self.params_coefs = params_coefs

        self.mean_genotypic_dist = 0
        self.stddev_genotypic_dist = 0

        self.mean_pgenotypic_dist_off = 0
        self.stddev_pgenotypic_dist_off = 0
        self.mean_pgenotypic_dist_ri = 0
        self.stddev_pgenotypic_dist_ri = 0
        self.mean_pgenotypic_dist_ei = 0
        self.stddev_pgenotypic_dist_ei = 0

        self.mean_pgenotypic_dist = 0
        self.stddev_pgenotypic_dist = 0

        self.mean_fitness, self.std_fitness = 0,0
        self.mean_fitness_test, self.std_fitness_test = 0,0
        self.best_fitness_test = 0,0
        
        self.mean_behavioural_diversity, self.std_behavioural_diversity = 0,0
        self.mean_pbehavioural_diversity, self.std_pbehavioural_diversity = 0,0
        
        self.mean_fitness_diversity, self.std_fitness_diversity = 0,0
        self.list_mean_fit = []
        self.list_mean_sizes = []

        self.mean_pop_size = 0
        self.std_pop_size = 0
        
        self.mean_n_functions = 0
        self.std_n_functions = 0
        self.mean_n_non_arith = 0
        self.std_n_non_arith = 0
        self.mean_max_seq = 0
        self.std_max_seq = 0
        self.mean_depth = 0
        self.std_depth = 0
        
        self.mean_pvalid_individuals = 0
        self.std_pvalid_individuals = 0

        self.indiv_keys = ['initialPopulation', 'eImmigrant', 'rImmigrant', 'cross', 'mut', 'crossMut']
        self.indiv_keys.sort()

        self.indiv_type_mean_fitness = {}
        self.indiv_type_std_fitness = {}
        self.indiv_type_mean_sizes = {}
        self.indiv_type_std_sizes = {}
        self.indiv_type = {}
        self.ancestors_type = {}
        for k in self.indiv_keys:
            self.indiv_type[k] = 0
            self.ancestors_type[k] = 0
            self.indiv_type_mean_fitness[k] = 0
            self.indiv_type_std_fitness[k] = 0
            self.indiv_type_mean_sizes[k] = 0
            self.indiv_type_std_sizes[k] = 0

        self.instances = instances
        self.evaluation_function = evaluation_function

        self.prob_type = prob_type
        if (self.prob_type == 'Boolean'):
            self.evaluation_type = evaluateNM_mat
            self.behavioural_distance = nm_behavioural_distance
        elif(self.prob_type == 'SR'):
            self.evaluation_type = evaluateSR_mat
            self.behavioural_distance = sr_behavioural_distance
        elif(self.prob_type == 'AgentController'):
            self.evaluation_type = evaluateAnt_mat
            self.behavioural_distance = ant_behavioural_distance
        elif(self.prob_type == 'Spambase'):
            self.evaluation_type = evaluateSpambase_mat
            self.behavioural_distance = Spambase_behavioural_distance
        elif(self.prob_type == 'LunarLander'):
            self.evaluation_type = evaluateLunarLander_mat
            self.behavioural_distance = LunarLander_behavioural_distance

        self.gsyngp_iterations = np.zeros((self.generations, self.pop_size))
        self.size_diff = np.zeros((self.generations, self.pop_size))
        self.ind_iterations = np.zeros(self.generations)
        self.ind_size_diff = np.zeros(self.generations)
        
    def evaluateGroup(self, inds):
        for i in range(len(inds)):
            inds[i].index = i
            self.evaluationQueue.put(inds[i])
        evaluated = inds[:]
        for j in range(len(inds)):
            ind = self.evaluatedQueue.get()
            evaluated[ind.index] = ind
        return evaluated

    def parallel_diversities_relativeToBest(self):
        #computes the diversities of the best individual regarding all others
        if (self.compute_diversities):
            genotypes = [np.array(ind.genotype) for ind in self.population]
            self.mean_genotypic_dist, self.stddev_genotypic_dist, self.mean_pgenotypic_dist, self.stddev_pgenotypic_dist, distances, p_distances = compute_parallel_genotypic_diversity_relative_best(self.best_indiv.genotype,
                genotypes, self.genotypicDiversityProcessQueue, self.genotypicDiversityProcessedQueue)

            behaviours = [np.array(ind.behaviour) for ind in self.population]
            self.mean_behavioural_diversity, self.std_behavioural_diversity, self.mean_pbehavioural_diversity, self.std_pbehavioural_diversity, behavioural_distances, pbehavioural_distances = compute_parallel_behavioural_diversity_relative_best(self.best_indiv.behaviour,
                behaviours, self.behaviouralDiversityProcessQueue, self.behaviouralDiversityProcessedQueue)

            for i in range(len(genotypes)):
                self.population[i].genotypic_distances = np.array(distances[i])
                self.population[i].pgenotypic_distances = np.array(p_distances[i])
                self.population[i].mean_genotypic_dist = np.mean(self.population[i].genotypic_distances)
                self.population[i].stddev_genotypic_dist = np.std(self.population[i].genotypic_distances)
                self.population[i].mean_pgenotypic_dist = np.mean(self.population[i].pgenotypic_distances)
                self.population[i].stddev_pgenotypic_dist = np.std(self.population[i].pgenotypic_distances)
                self.population[i].behavioural_distances = np.array(behavioural_distances[i])
                self.population[i].pbehavioural_distances = np.array(pbehavioural_distances[i])
                self.population[i].mean_behavioural_dist = np.mean(self.population[i].behavioural_distances)
                self.population[i].stddev_behavioural_dist = np.std(self.population[i].behavioural_distances)
                self.population[i].mean_pbehavioural_dist = np.mean(self.population[i].pbehavioural_distances)
                self.population[i].stddev_pbehavioural_dist = np.std(self.population[i].pbehavioural_distances)

            self.best_indiv.genotypic_distances = distances
            self.best_indiv.pgenotypic_distances = p_distances
            self.best_indiv.mean_genotypic_dist = np.mean(self.best_indiv.genotypic_distances)
            self.best_indiv.stddev_genotypic_dist = np.std(self.best_indiv.genotypic_distances)
            self.best_indiv.mean_pgenotypic_dist = np.mean(self.best_indiv.pgenotypic_distances)
            self.best_indiv.stddev_pgenotypic_dist = np.std(self.best_indiv.pgenotypic_distances)
            self.best_indiv.behavioural_distances = behavioural_distances
            self.best_indiv.pbehavioural_distances = pbehavioural_distances
            self.best_indiv.mean_behavioural_dist = np.mean(self.best_indiv.behavioural_distances)
            self.best_indiv.stddev_behavioural_dist = np.std(self.best_indiv.behavioural_distances)
            self.best_indiv.mean_pbehavioural_dist = np.mean(self.best_indiv.pbehavioural_distances)
            self.best_indiv.stddev_pbehavioural_dist = np.std(self.best_indiv.pbehavioural_distances)
            
            self.mean_fitness, self.std_fitness = compute_fitness_diversity(np.array([ind.fitness for ind in self.population]))

        if (self.compute_gp_metrics):
            executed_nodes = np.array([len(self.population[i].executedNodes)*1.0 for i in range(len(self.population))])
            self.mean_executed_nodes = np.mean(executed_nodes)
            self.std_executed_nodes = np.std(executed_nodes)

            pop_size = np.array([len(self.population[i].genotype) for i in range(len(self.population))])  # [len(self.population[i].executedNodes) * 1.0 / len(self.population[i].genotype) for i in range(len(self.population))]
            pexecuted_nodes = executed_nodes / pop_size
            self.mean_percentage_executed_nodes = np.mean(pexecuted_nodes)
            self.std_percentage_executed_nodes = np.std(pexecuted_nodes)

            self.mean_pop_size = np.mean(pop_size)
            self.std_pop_size = np.std(pop_size)
       
    class node:
        def __init__(self, v, s, depth):
            self.value = v
            self.sons = s
            self.depth = depth
            
    def to_tree(self, function_mat, terminal_mat, ind, index=0):
        if index >= len(ind):
            return None, index, 0
        val = ind[index]
        sym = val.rstrip('()')
        if sym in function_mat:
            arity = function_mat[sym]['arity']
            sons = []
            curr_index = index + 1
            max_child_depth = 0
            for _ in range(arity):
                child, curr_index, child_max_depth = self.to_tree(function_mat, terminal_mat, ind, curr_index)
                if child is None:
                    return None, index, 0
                sons.append(child)
                max_child_depth = max(max_child_depth, child_max_depth)
            my_depth = max_child_depth + 1
            n = self.node(val, sons, my_depth)
            return n, curr_index, my_depth
        else:
            leaf = self.node(val, None, 0)
            return leaf, index + 1, 0
    
    def ind_interpretability(self, function_set, function_mat, terminal_mat):
            
        results = []
        arithmetic_set = set(['add', 'sub', 'mult', 'div'])
        non_arith = set(function_set) - arithmetic_set
        
        for i in range(len(self.population)):
            genotype = self.population[i].genotype
            
            n_functions = sum(1 for symbol in genotype if symbol.split('(')[0] in function_set)
            n_non_arith = sum(1 for symbol in genotype if symbol.split('(')[0] in non_arith)
                 
            max_seq = 0
            curr_seq = 0
            for symbol in genotype:
                if symbol.split('(')[0] in non_arith:
                    curr_seq += 1
                    max_seq = max(max_seq, curr_seq)
                else:
                    curr_seq = 0

            max_depth = self.to_tree(function_mat, terminal_mat, genotype, index=0)[2]
        
            results.append({
                'n_functions': n_functions,
                'n_non_arithmetic': n_non_arith,
                'max_non_arith_seq': max_seq,
                'max_depth': max_depth,
            })
                    
        self.mean_n_functions = np.mean([res['n_functions'] for res in results])
        self.std_n_functions = np.std([res['n_functions'] for res in results])
        self.mean_n_non_arith = np.mean([res['n_non_arithmetic'] for res in results])
        self.std_n_non_arith = np.std([res['n_non_arithmetic'] for res in results])
        self.mean_max_seq = np.mean([res['max_non_arith_seq'] for res in results])
        self.std_max_seq = np.std([res['max_non_arith_seq'] for res in results])
        self.mean_depth = np.mean([res['max_depth'] for res in results])
        self.std_depth = np.std([res['max_depth'] for res in results])
    

    # 2025 Inssersion of control prints
    def pop_based_evolution(self, pop_initialization, crossovers, crossover_odds, mutations, mutation_odds,
                            parent_selection, survivors_selection, stage_log, pop=None, start_gen=0, randomState=None, metrics=None):
        
        ##paralelizar avaliacao
        ev = eval(self.evaluation_function)
        ev.initialize()
        
        self.evaluationQueue, self.evaluatedQueue = Queue(), Queue()
        self.evaluation_processes = []
        for i in range(self.instances):
            if self.prob_type == "Boolean":
                self.evaluation_processes.append(Process(target=self.evaluation_type, args=(self.evaluationQueue, self.evaluatedQueue,  self.evaluation_function)))
            elif(self.prob_type == 'SR'):
                self.evaluation_processes.append(Process(target=self.evaluation_type, args=(self.evaluationQueue, self.evaluatedQueue, self.evaluation_function)))   
            elif (self.prob_type == 'AgentController'):
                self.evaluation_processes.append(Process(target=self.evaluation_type, args=(self.evaluationQueue, self.evaluatedQueue, self.evaluation_function)))  
            elif (self.prob_type == 'Spambase'):
                self.evaluation_processes.append(Process(target=self.evaluation_type, args=(self.evaluationQueue, self.evaluatedQueue, self.evaluation_function)))
            elif (self.prob_type == 'LunarLander'):
                self.evaluation_processes.append(Process(target=self.evaluation_type, args=(self.evaluationQueue, self.evaluatedQueue, self.evaluation_function)))
            self.evaluation_processes[-1].start()

        ##paralelizar reproducao
        self.parentQueue = Queue()
        self.offspringQueue = Queue()
        self.crossover_processes = []
        for i in range(self.instances):
            self.crossover_processes.append(Process(target=parallel_reproduction, args=(self.parentQueue, self.offspringQueue, self.p_cross, self.p_mut, crossovers, crossover_odds, mutations, mutation_odds)))
            self.crossover_processes[-1].start()

        ##paralelizar genotypic diversity
        self.genotypicDiversityProcessQueue = Queue()
        self.genotypicDiversityProcessedQueue = Queue()
        self.genotypic_diversity_processes = []
        for i in range(self.instances):
            self.genotypic_diversity_processes.append(Process(target=pairwise_genotypic_diversity, args=(self.genotypic_distance, self.genotypicDiversityProcessQueue, self.genotypicDiversityProcessedQueue)))
            self.genotypic_diversity_processes[-1].start()
            
        ##paralelizar behavioural diversity
        self.behaviouralDiversityProcessQueue = Queue()
        self.behaviouralDiversityProcessedQueue = Queue()
        self.behavioural_diversity_processes = []
        for i in range(self.instances):
            self.behavioural_diversity_processes.append(Process(target=pairwise_behavioural_diversity, args=(self.behavioural_distance, self.behaviouralDiversityProcessQueue, self.behaviouralDiversityProcessedQueue)))
            self.behavioural_diversity_processes[-1].start()

        ##criar pop inicial ou restaurar a partir de ponto de gravacao
        if(pop is None):
            self.population = pop_initialization(self.pop_size)
            evaluated = self.evaluateGroup(self.population)
            for j in range(self.pop_size):
                evaluated[j].label = 'initialPopulation'
                evaluated[j].creation=-1
            self.population = evaluated

            for j in range(self.pop_size):
                if (self.best_indiv == None or (not self.maximise and self.population[j].fitness < self.best_indiv.fitness) or (
                        self.maximise and self.population[j].fitness > self.best_indiv.fitness)):
                    self.best_indiv = self.population[j]

            self.sort_list(self.population)

            print('best of initial pop : ' + str(
                self.best_indiv.fitness) + ', vals: ' + str(
                self.best_indiv.fitness_vals), self.best_indiv.genotype)

            self.parallel_diversities_relativeToBest()

            self.saveState(-1)

        else:
            self.population = pop
            random.setstate(randomState)
            keys = list(metrics.keys())
            for k in keys:
                exec('self.'+k+'= metrics[\'' + k + '\']')

            self.population = self.evaluateGroup(self.population)
            for j in range(self.pop_size):
                if (self.best_indiv == None or (not self.maximise and self.population[j].fitness < self.best_indiv.fitness) or (
                        self.maximise and self.population[j].fitness > self.best_indiv.fitness)):
                    self.best_indiv = self.population[j]
                
        for gen in range(start_gen,self.generations):
            if random.random() < self.p_reeval:
                self.population = self.evaluateGroup(self.population)
                for j in range(self.pop_size):
                    if (self.best_indiv == None or (
                            not self.maximise and self.population[j].fitness < self.best_indiv.fitness) or (
                            self.maximise and self.population[j].fitness > self.best_indiv.fitness)):
                        self.best_indiv = self.population[j]

                self.sort_list(self.population)

            offsprings = []
            if self.ri > 0:
                immigrants = pop_initialization(self.ri)
            else:
                immigrants = []
            if self.ei > 0:
                immigrants.extend([self.elitist_immigrant(parent_selection, mutations, mutation_odds) for i in range(self.ei)])

            for i in range(self.ei + self.ri):
                if i < self.ri:
                    immigrants[i].label = 'rImmigrant'
                else:
                    immigrants[i].label = 'eImmigrant'
                immigrants[i].creation = gen

            offsprings.extend(immigrants)

            o = 0
            while(o<self.n_offsprings):
                parentA = parent_selection()
                parentB = parent_selection()

                self.parentQueue.put((parentA, parentB, gen))
                o+=1

            for o in range(self.n_offsprings):
                (off, iterations_performed, size_diff) = self.offspringQueue.get()    
                if('cross' in off.label):
                    for it in iterations_performed:
                        self.ind_iterations[gen]+=1
                    for sd in size_diff:
                        self.size_diff[gen][int(self.ind_size_diff[gen])] = sd
                        self.ind_size_diff[gen]+=1
                off.creation = gen
                offsprings.append(off)
                
            offsprings = self.evaluateGroup(offsprings)
            survivors_selection(offsprings)
            
            for j in range(self.pop_size):
                if (self.best_indiv == None or (
                        not self.maximise and self.population[j].fitness < self.best_indiv.fitness) or (
                        self.maximise and self.population[j].fitness > self.best_indiv.fitness)):
                    self.best_indiv = self.population[j]            
            
            self.parallel_diversities_relativeToBest()

            self.p_cross*=self.params_coefs[0]
            self.p_mut*=self.params_coefs[1]
            
            self.ind_interpretability(self.function_set, self.function_mat, self.terminal_mat)

            stage_log(gen)
            self.saveState(gen)
        
        self.best_fitness_test = eval(self.evaluation_function).evaluate_test(self.best_indiv)
        list_mean_fit_test = []
        for i in self.population:
            list_mean_fit_test.append(eval(self.evaluation_function).evaluate_test(i))
        self.mean_fitness_test = np.mean(list_mean_fit_test)
        self.std_fitness_test = np.std(list_mean_fit_test)
        
        stage_log(self.generations)
        self.saveState(self.generations)
        
        for i in range(self.instances*2):
            self.parentQueue.put(None)
            self.evaluationQueue.put(None)
            self.genotypicDiversityProcessQueue.put(None)
       
        for i in range(self.instances):
            self.evaluation_processes[i].join()
            self.crossover_processes[i].join()
            self.genotypic_diversity_processes[i].join()

    def sort_list(self, l):
        for i in range(len(l)):
            j = i + 1
            while (j < len(l)):
                if (i != j and ((not self.maximise and l[i].fitness > l[j].fitness) or (self.maximise and l[i].fitness < l[j].fitness))):
                    a = l[i]
                    l[i] = l[j]
                    l[j] = a
                j += 1

    def tournament_selection(self):
        champ = random.choice(self.population)
        for i in range(self.tourn_size - 1):
            a = random.choice(self.population)

            if ((self.maximise and champ.fitness < a.fitness) or (not self.maximise and champ.fitness > a.fitness)):
                champ = a
        return champ
    
    def clearing_tournament_selection_mat(self):
        unique_candidates = {}
        for ind in self.population:
            genotype_str = str(ind.genotype)
            if genotype_str not in unique_candidates:
               unique_candidates[genotype_str] = ind
        tournament = [random.choice(unique_candidates.values()) for _ in range(self.tourn_size)]
        if self.maximise:
            champ = max(tournament, key=lambda ind: ind.fitness)
        else:
            champ = min(tournament, key=lambda ind: ind.fitness)       
        return champ
    
    def rolette_wheel_selection(self):
        sum_fitness = 0.0
        probs = []
        for ind in self.population:
            probs.append(ind.fitness)
            sum_fitness += ind.fitness
        for i in range(len(probs)):
            probs[i] /= sum_fitness
        if(not self.maximise):
            for i in range(len(probs)):
                probs[i] = 1.0-probs[i]
        r = random.random()
        accumulated = 0
        for i in range(len(probs)):
            accumulated += probs[i]
            if (r < accumulated):
                return self.population[i]
        return self.population[-1]

    def elitist_immigrant(self, parent_selection, mutations, odds):
        off = self.best_indiv.copy_self()
        r = random.random()
        for i in range(len(odds)):
            if r < odds[i]:
                off = mutations[i](off)
                break
        off.make_genotype()
        return off

    def merge_selection(self, offs):
        self.population.extend(offs)
        self.sort_list(self.population)
        self.population = self.population[:self.pop_size]


    def elitist_selection(self, offs):
        self.sort_list(self.population)
        self.population = self.population[:self.elite_size]
        self.sort_list(offs)
        self.population.extend(offs[:self.pop_size - self.elite_size])
        self.sort_list(self.population)

    def saveState(self, gen):
        l = self.log_file.split('.')
        i = self.log_file.index(l[-1]) - 1
        fname = self.log_file[:i] + '_gen' + str(gen) + '.txt'
        f = open(fname, 'w')
        f.write('<randomState>' + str(random.getstate()) + '</randomState>\n')
        for ind in self.population:
            f.write(ind.toStringComplete() + '\n')

        f.write('<mean_genotypic_dist>' + str(self.mean_genotypic_dist) + '</mean_genotypic_dist>\n')
        f.write('<stddev_genotypic_dist>' + str(self.stddev_genotypic_dist) + '</stddev_genotypic_dist>\n')
        f.write('<mean_pgenotypic_dist_off>' + str(self.mean_pgenotypic_dist_off) + '</mean_pgenotypic_dist_off>\n')
        f.write('<stddev_pgenotypic_dist_off>' + str(self.stddev_pgenotypic_dist_off) + '</stddev_pgenotypic_dist_off>\n')
        f.write('<mean_pgenotypic_dist_ri>' + str(self.mean_pgenotypic_dist_ri) + '</mean_pgenotypic_dist_ri>\n')
        f.write('<stddev_pgenotypic_dist_ri>' + str(self.stddev_pgenotypic_dist_ri) + '</stddev_pgenotypic_dist_ri>\n')
        f.write('<mean_pgenotypic_dist_ei>' + str(self.mean_pgenotypic_dist_ei) + '</mean_pgenotypic_dist_ei>\n')
        f.write('<stddev_pgenotypic_dist_ei>' + str(self.stddev_pgenotypic_dist_ei) + '</stddev_pgenotypic_dist_ei>\n')
        f.write('<mean_pgenotypic_dist>' + str(self.mean_pgenotypic_dist) + '</mean_pgenotypic_dist>\n')
        f.write('<stddev_pgenotypic_dist>' + str(self.stddev_pgenotypic_dist) + '</stddev_pgenotypic_dist>\n')
        f.write('<mean_behavioural_dist>' + str(self.mean_behavioural_diversity) + '</mean_behavioural_dist>\n')
        f.write('<stddev_behavioural_dist>' + str(self.std_behavioural_diversity) + '</stddev_behavioural_dist>\n')
        f.write('<mean_pop_size>' + str(self.mean_pop_size) + '</mean_pop_size>\n')
        f.write('<std_pop_size>' + str(self.std_pop_size) + '</std_pop_size>\n')

        f.close()

        if (gen > 1):
            fname = self.log_file[:i] + '_gen' + str(gen - 2) + '.txt'
            if (os.path.exists(fname)):
                os.remove(fname)

    def restoreState(self, gp=True, function_set=None):
        l = self.log_file.split('.')
        i = self.log_file.index(l[-1]) - 1
        gen = self.generations-1
        while(gen>=0):
            fname = self.log_file[:i] + '_gen' + str(gen) + '.txt'
            try:
                f = open(fname, 'r')
                randomState = f.readline()
                if(not ('<randomState>' in randomState and '</randomState>' in randomState)):
                    gen-=1
                    continue
                randomState = randomState[randomState.index('>')+1:]
                randomState = eval(randomState[:randomState.index('<')])
                pop = []
                line = f.readline()
                while('<ind>' in line):

                    ind = {}
                    line = f.readline()
                    while not '</ind>' in line:
                        key = line[1:line.index('>')]
                        val = line[line.index('>')+1:]
                        val = val[:val.index('</')]
                        try:
                            val = eval(val)
                            if key == 'genotype':
                                val = np.array(val,dtype = '<U100')
                            elif isinstance(val, list):
                                val = np.array(val)
                            ind[key] = val
                            ind[key] = val
                        except:

                            ind[key] = None
                        line = f.readline()
                    pop.append(ind)

                    line = f.readline()
                metrics = {}
                while(line!=''):
                    metric = line[1:line.index('>')]
                    line = line[line.index('>')+1:]
                    line = line[:line.index('<')]
                    metrics[metric] = float(line)
                    line = f.readline()
                f.close()
                return [gen,randomState, pop, metrics]
            except FileNotFoundError as e:
                pass
            gen-=1
        return None


def evaluateAnt_mat(evaluationQueue, evaluatedQueue, evaluationFunction):
    ev = eval(evaluationFunction)
    evaluation_function = ev.evaluate
    while(True):
        ind = evaluationQueue.get()
        if(ind is None):
            break
        ind.add_fitness(evaluation_function(ind))
        evaluatedQueue.put(ind)

def evaluateSR_mat(evaluationQueue, evaluatedQueue, evaluationFunction):
    ev = eval(evaluationFunction)
    evaluation_function = ev.evaluate
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind.add_fitness(evaluation_function(ind))
        evaluatedQueue.put(ind)

# 2025 Evaluate function to N-Multiplexer benchmark problems
def evaluateNM_mat(evaluationQueue, evaluatedQueue, evaluationFunction):
    ev = eval(evaluationFunction)
    evaluation_function = ev.evaluate
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind.add_fitness(evaluation_function(ind))
        evaluatedQueue.put(ind)
        
def evaluateSpambase_mat(evaluationQueue, evaluatedQueue, evaluationFunction):
    ev = eval(evaluationFunction)
    evaluation_function = ev.evaluate
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind.add_fitness(evaluation_function(ind))
        evaluatedQueue.put(ind)

def evaluateLunarLander_mat(evaluationQueue, evaluatedQueue, evaluationFunction):
    ev = eval(evaluationFunction)
    evaluation_function = ev.evaluate
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind.add_fitness(evaluation_function(ind))
        evaluatedQueue.put(ind)

def pairwise_genotypic_diversity(genotypic_distance, genotypicDiversityProcessQueue, genotypicDiversityProcessedQueue):

    task = genotypicDiversityProcessQueue.get()
    while(task is not None):
        indB = task[2]
        indA = task[1]
        ind=task[0]

        a, b = genotypic_distance(indA, indB)

        genotypicDiversityProcessedQueue.put((ind, a, a*1.0/len(indA)))

        task = genotypicDiversityProcessQueue.get()

def compute_parallel_genotypic_diversity_relative_best(best_genotype, genotypes, genotypicDiversityProcessQueue,genotypicDiversityProcessedQueue):
    genotypic_distances = np.zeros(len(genotypes))
    pgenotypic_distances = np.zeros(len(genotypes))

    for ind in range(len(genotypes)):
        genotypicDiversityProcessQueue.put((ind, best_genotype, genotypes[ind]))

    for ind in range(len(genotypes)):
        res = genotypicDiversityProcessedQueue.get()
        index = res[0]
        genotypic_distances[index] = res[1]
        pgenotypic_distances[index] = res[2]

    return np.mean(genotypic_distances), np.std(genotypic_distances), np.mean(pgenotypic_distances), np.std(pgenotypic_distances), genotypic_distances, pgenotypic_distances

def pairwise_behavioural_diversity(behavioural_distance, behaviouralDiversityProcessQueue,behaviouralDiversityProcessedQueue):

    task = behaviouralDiversityProcessQueue.get()
    while(task is not None):
        behaviourB = task[2]
        behaviourA = task[1]
        ind = task[0]

        a = behavioural_distance(behaviourA, behaviourB)
        if(not np.isfinite(a)):
            a=0
        if len(behaviourA) == 0:
            behaviouralDiversityProcessedQueue.put((ind, 0, 0))
        else:
            behaviouralDiversityProcessedQueue.put((ind, a, a * 1.0 / len(behaviourA)))

        task = behaviouralDiversityProcessQueue.get()

def compute_parallel_behavioural_diversity_relative_best(best_behaviour, behaviours, behaviouralDiversityProcessQueue,behaviouralDiversityProcessedQueue):
    behavioural_distances = np.zeros(len(behaviours))
    pbehavioural_distances = np.zeros(len(behaviours))

    for ind in range(len(behaviours)):
        behaviouralDiversityProcessQueue.put((ind, best_behaviour, behaviours[ind]))

    for ind in range(len(behaviours)):
        res = behaviouralDiversityProcessedQueue.get()
        index = res[0]
        behavioural_distances[index] = res[1]
        pbehavioural_distances[index] = res[2]

    return np.mean(behavioural_distances), np.std(behavioural_distances), np.mean(pbehavioural_distances), np.std(pbehavioural_distances), behavioural_distances, pbehavioural_distances

def sr_behavioural_distance(bA, bB):
    return np.sqrt(np.sum((bB-bA)**2))

def ant_behavioural_distance(bA, bB):
    return np.sqrt(np.sum((bB-bA)**2))

def nm_behavioural_distance(bA, bB):
    if np.isnan(bA).any() or np.isnan(bB).any():
        return 1.0 
    if len(bA) == 0:
        return 0.0
    return np.sum(bA != bB) / len(bA)

def Spambase_behavioural_distance(bA, bB):
    return np.sqrt(np.mean((bA - bB)**2))

def LunarLander_behavioural_distance(bA, bB):
    # Normalized Hamming distance for action sequences
    min_len = min(len(bA), len(bB))
    diff = np.sum(np.array(bA[:min_len]) != np.array(bB[:min_len]))
    if min_len == 0:
        return 0.0
    return diff / min_len

def parallel_reproduction(parentQueue, offspringQueue, p_cross, p_mut, crossovers, crossover_odds, mutations, mutation_odds):
    random.seed(time.time())

    parents = parentQueue.get()
    while(parents is not None):
        iterations_performed=[]
        size_diff=[]
        parentA, parentB, gen = parents[0], parents[1], parents[2]
        crossed = False
        if random.random() < p_cross:
            r = random.random()
            for i in range(len(crossovers)):
                if r < crossover_odds[i]:
                    offs, iterations_performed, size_diff = crossovers[i](parentA, parentB, 0, 1, gen)
                    for off in offs:
                        off.label = 'cross'
                    crossed = True
                    break
        else:
            offs = [parentA.copy_self()]
            offs[0].label = 'mut'

        if not crossed or random.random() < p_mut:
            for k in range(len(offs)):
                r = random.random()
                for i in range(len(mutations)):
                    if r < mutation_odds[i]:
                        offs[k] = mutations[i](offs[k])
                        if crossed:
                            offs[k].label = 'crossMut'
                        break
        for off in offs:
            off.make_genotype()
            offspringQueue.put((off, iterations_performed, size_diff))

        parents = parentQueue.get()

def compute_fitness_diversity(fits):

    end = len(fits)
    for f in range(len(fits)):

        if(np.isinf(fits[f])):
            end = f
            break

    fits = np.array(list(fits)[:end])
    return np.mean(fits), np.std(fits)
