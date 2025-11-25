
import os
import gymnasium as gym

def read_config(filename):
    f = open(filename)
    line = f.readline()
    params={}
    while(line!=''):
        if('\n' in line):
            line = line[:-1]
        line = line.split('\t')
        params[line[0]] = eval(line[1])
        line = f.readline()
    f.close()
    return params

# 2025 write_config_file -> write all in one folder
def write_config_file(configs, suffix):
    config_dir = 'config_files'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    sel = "tourn" + str(configs["tourn_size"])
    fname = (f"{configs['problem']}_{configs['strat_name']}_{configs['iteration']}_"
             f"{configs['ri']}ri{configs['ei']}ei_{configs['p_cross']}pcross"
             f"{configs['p_mut']}pmut_{sel}{suffix}.txt")
    
    full_path = os.path.join(config_dir, fname)
    
    with open(full_path, 'w') as f:
        for k in configs.keys():
            val = configs[k]
            if (isinstance(val, str)):
                val = '\"'+val+'\"'
            else: 
                val = str(val)
            f.write(k+'\t'+val+'\n') 
        
# 2025 make_config_Spambase_mat -> new benchmark Spambase     
def make_config_Spambase_mat(parent_selection):    
    algs = ['GSynGP', 'STGP']
    problems = ['Spambase1']
    iterations = ['1']
    immigrants = [(0.0, 0.0)]
    variation_operators = [(0.7,0.3)]
    
    data_types = ['float', 'bool']    
                
    function_set = ['add', 'sub', 'mult', 'div', 'and', 'or', 'not', 'lt', 'eq', 'if-then-else']
    function_mat = {
        'add': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'sub': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'mult': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'div': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'and': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'or': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'not': {'arity': 1, 'args': ['bool'], 'return': 'bool'},
        'lt': {'arity': 2, 'args': ['float', 'float'], 'return': 'bool'},
        'eq': {'arity': 2, 'args': ['float', 'float'], 'return': 'bool'},
        'if-then-else': {'arity': 3, 'args': ['bool', 'float', 'float'], 'return': 'float'}
    }
    
    terminal_set = ['bit', 'var', 'uniform-constant']

    terminal_mat = {
        'bit': {'type': 'bool'},
        'var': {'type': 'float'},
        'uniform-constant': {'type': 'float'}
    }
    
    terminal_params = {
        'uniform-constant': {'value':"np.linspace(0,100,100)"},
        'bit': {'value': "[0,1]"}
    }
    
    suffix = '_spambase'
    log_directory = 'logs' + suffix
    
    prob_type = 'Spambase'
        
    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP',
    'restore': False, 'optimize_strat': True, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6, 'max_depth': 17,
    'p_reeval': 0.0, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': '1', 'pop_size': 250, 'generations': 500,
    'ri': 0.0, 'ei':0.0,
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'Spambase1', 'prob_type': prob_type, 
    'function_set': function_set,'function_mat': function_mat, 'terminal_set': terminal_set, 'terminal_mat': terminal_mat, 'data_types': data_types,
    'terminal_params': terminal_params, 'parent_selection': parent_selection}    

    for prob in problems:
        configs['problem'] = prob
        ndimensions = 57
        domain = [[0, 100] for i in range(ndimensions)]
        npoints = 400
    
        configs['ndimensions'] = ndimensions
        configs['domain'] = domain
        configs['npoints'] = npoints
    
        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    for ri, ei in immigrants:
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut
                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs,suffix)

# 2025 make_configs_sr_ma -> functions arities parametrize
def make_configs_sr_mat(arithmetic, consts, parent_selection):
    algs = ['GSynGP', 'STGP']
    problems = ['Koza1','Paige1', 'Keijzer12', 'Koza3']
    iterations = ['1']
    immigrants = [(0.07,0.03)] 
    variation_operators = [(0.7,0.3)]    
    
    data_types = ['float']
    
    if(arithmetic):    
        function_set = ['add', 'sub', 'mult', 'div'] 
        function_mat = {
            'add': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'sub': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'mult': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'div': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'}
        }
    else:
        function_set = ['add', 'sub', 'mult', 'div', 'sin', 'cos', 'exp', 'lnmod']
        function_mat = {
            'add': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'sub': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'mult': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'div': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
            'sin': {'arity': 1, 'args': ['float'], 'return': 'float'},
            'cos': {'arity': 1, 'args': ['float'], 'return': 'float'},
            'exp': {'arity': 1, 'args': ['float'], 'return': 'float'},
            'lnmod': {'arity': 1, 'args': ['float'], 'return': 'float'}
        }

    if(consts):
        terminal_set = ['var','uniform-constant']
        terminal_mat = {
            'var': {'type': 'float'},
            'uniform-constant': {'type': 'float'}
        }
    else:
        terminal_set = ['var']
        terminal_mat = {
            'var': {'type': 'float'}
        }
        
    suffix = ''
    if(arithmetic):
        suffix += '_arithmetic'
    if(consts):
        suffix+='_consts'

    log_directory = 'logs' + suffix

    prob_type = 'SR'

    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP',
    'restore': False, 'optimize_strat': True, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6,'max_depth': 17,
    'p_reeval': 0.0, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': '1', 'pop_size': 250, 'generations': 500,
    'ri': 0.0, 'ei':0.0,
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'Koza1', 'prob_type': prob_type, 
    'function_set': function_set,'function_mat': function_mat, 'terminal_set': terminal_set, 'terminal_mat': terminal_mat, 'data_types': data_types,
    'terminal_params':{'uniform-constant':{'value':"np.linspace(-1,1,21)"}}, 'parent_selection': parent_selection}

    for prob in problems:
        configs['problem'] = prob
        if('Koza' in prob or 'Nguyen5' in prob):
            ndimensions=1
            domain = [[-1,1] for i in range(ndimensions)]
            npoints = 20

        elif(prob == 'Paige1'):
            ndimensions=2
            domain = [[-5,5] for i in range(ndimensions)]
            npoints = 20
        elif(prob=='Keijzer12'):
            ndimensions=2
            domain = [[-3,3] for i in range(ndimensions)]
            npoints = 20            
        else:    
            ndimensions=2
            domain = [[-10,10] for i in range(ndimensions)]
            npoints = 20

        configs['ndimensions'] = ndimensions
        configs['domain'] = domain
        configs['npoints'] = npoints
        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    print(alg, it)
                    for ri, ei in immigrants:                     
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut
                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs,suffix)

# 2025 benchmark Santa Fe Ant Trail cause Santa Fe Ant Trail naturally use difrent arities
def make_configs_sf_mat(parent_selection):
    algs = ['GSynGP', 'STGP']
    problems = ['SantaFeAntTrailMin', 'LosAltosHillsAntTrailMin']
    iterations = ['1']
    immigrants = [(0.0,0.0),(0.07,0.03)]
    variation_operators = [(0.7,0.3)]

    domain = [[-1,1]]
    ndimensions = 1
    npoints = 20
    prob_type = 'AgentController'

    suffix = ''
    log_directory = 'logs' + suffix
    
    data_types = ['action']

    function_set = ['iffoodAhead', 'Progn2', 'Progn3']
    
    function_mat = {
        'iffoodAhead': {'arity': 2,'args': ['action', 'action'], 'return': 'action'},
        'Progn2': {'arity': 2, 'args': ['action', 'action'], 'return': 'action'},
        'Progn3': {'arity': 3, 'args': ['action', 'action', 'action'], 'return': 'action'}
    }

    terminal_set = ['left', 'right', 'move']
    terminal_mat = {
        'left': {'type': 'action'},
        'right': {'type': 'action'},
        'move': {'type': 'action'}
    }
    

    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP',
    'restore': False, 'optimize_strat': True, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6,'max_depth': 17,
    'p_reeval': 0.0, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': '1', 'pop_size': 250, 'generations': 500,
    'ri': 0.0, 'ei':0.0, 
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'SantaFeAntTrailMin', 'prob_type': prob_type, 
    'domain': domain, 'ndimensions': ndimensions, 'npoints': npoints, 
    'function_set': function_set, 'function_mat': function_mat, 'terminal_set': terminal_set, 'terminal_mat': terminal_mat, 'data_types': data_types,
    'terminal_params':{}, 'parent_selection': parent_selection}

    for prob in problems:
        configs['problem'] = prob

        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    print(alg, it)
                    for ri, ei in immigrants:                     
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut

                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs, suffix)
                                
# 2025 new benchmark N-Multiplexer problems cause N-Multiplexer naturally use difrent arities                                
def make_configs_NMultiplexer_mat(parent_selection):
    algs = ['GSynGP', 'STGP']
    problems = ['NMultiplexer3','NMultiplexer6', 'NMultiplexer11', 'NMultiplexer20']
    iterations = ['1']
    immigrants = [(0.0, 0.0)]
    variation_operators = [(0.7, 0.3)]
    
    data_types = ['bool']    
    
    function_set = ['if-then-else-boolean', 'and', 'or', 'not']
    function_mat = {
        'if-then-else-boolean': {'arity': 3, 'args': ['bool', 'bool', 'bool'], 'return': 'bool'},
        'and': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'or': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'not': {'arity': 1, 'args': ['bool'], 'return': 'bool'}
    }
        
    terminal_set = ['bit']
    terminal_mat = {
        'bit': {'type': 'bool'}
    }
    
    suffix = '_nmult'
    log_directory = 'logs' + suffix
    prob_type = 'Boolean'
    
    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP',
    'restore': False, 'optimize_strat': True, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'],
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth': 6, 'max_depth': 17,
    'p_reeval': 0.0, 'params_coefs': (1, 1),
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': '1', 'pop_size': 250, 'generations': 500,
    'ri': 0.0, 'ei': 0.0,
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353],
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'NMultiplexer3', 'prob_type': prob_type,
    'function_set': function_set, 'function_mat': function_mat, 'terminal_set': terminal_set, 'terminal_mat': terminal_mat, 'data_types': data_types,
    'terminal_params': {}, 'parent_selection': parent_selection}    

    for prob in problems:
        configs['problem'] = prob
        if prob == 'NMultiplexer3':
            ndimensions = 3
            domain = [[0, 1] for i in range(ndimensions)]
            npoints = 2 ** ndimensions

        elif prob == 'NMultiplexer6':
            ndimensions = 6
            domain = [[0, 1] for i in range(ndimensions)]
            npoints = 2 ** ndimensions
            
        elif prob == 'NMultiplexer11':
            ndimensions = 11
            domain = [[0, 1] for i in range(ndimensions)]
            npoints = 2 ** ndimensions
            
        elif prob == 'NMultiplexer20':
            ndimensions = 20
            domain = [[0, 1] for i in range(ndimensions)]
            npoints = 2 ** ndimensions
        
        configs['ndimensions'] = ndimensions
        configs['domain'] = domain
        configs['npoints'] = npoints

        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    for ri, ei in immigrants:                     
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut

                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs,suffix)

# 2025 new benchmark LunarLander cause LunarLander naturally use difrent arities
def make_config_LunarLander(parent_selection):    
    algs = ['GSynGP', 'STGP']
    problems = ['LunarLander1']
    iterations = ['1']
    immigrants = [(0.07,0.03)] 
    variation_operators = [(0.7,0.3)]

    data_types = ['float', 'bool', 'action']

    function_set = ['if-then-else-action', 'if-then-else', 'add', 'sub', 'mult', 'div', 'eq', 'lt', 'gt', 'cos', 'sin', 'tan', 'and', 'or', 'not']

    function_mat = {
        'if-then-else-action': {'arity': 3, 'args': ['bool', 'action', 'action'], 'return': 'action'},
        'if-then-else': {'arity': 3, 'args': ['bool', 'float', 'float'], 'return': 'float'},
        'add': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'sub': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'mult': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'div': {'arity': 2, 'args': ['float', 'float'], 'return': 'float'},
        'eq': {'arity': 2, 'args': ['float', 'float'], 'return': 'bool'},
        'lt': {'arity': 2, 'args': ['float', 'float'], 'return': 'bool'},
        'gt' : {'arity': 2, 'args': ['float', 'float'], 'return': 'bool'},
        'cos': {'arity': 1, 'args': ['float'], 'return': 'float'},
        'sin': {'arity': 1, 'args': ['float'], 'return': 'float'},
        'tan': {'arity': 1, 'args': ['float'], 'return': 'float'},
        'and': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'or': {'arity': 2, 'args': ['bool', 'bool'], 'return': 'bool'},
        'not': {'arity': 1, 'args': ['bool'], 'return': 'bool'}
    }
        
    terminal_set = ['x-cord', 'y-cord', 'x-velocity', 'y-velocity', 'angle', 'angular-velocity', 'left-leg-contact', 'right-leg-contact', 'uniform-constant', 'nothing', 'fire-action', 'left-action', 'right-action']

    terminal_mat = {
        'x-cord': {'type': 'float'},
        'y-cord': {'type': 'float'},
        'x-velocity': {'type': 'float'},
        'y-velocity': {'type': 'float'},
        'angle': {'type': 'float'},
        'angular-velocity': {'type': 'float'},
        'left-leg-contact': {'type': 'bool'},
        'right-leg-contact': {'type': 'bool'},
        'uniform-constant': {'type': 'float'},
        'nothing': {'type': 'action'},
        'fire-action': {'type': 'action'},
        'left-action': {'type': 'action'},
        'right-action': {'type': 'action'},
    }

    terminal_params = {
        'uniform-constant': {'value':"np.linspace(-1, 1, 100)"},
    }

    suffix = '_LunarLander'
    log_directory = 'logs' + suffix
    prob_type = 'LunarLander'

    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP',
    'restore': False, 'optimize_strat': True, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6, 'max_depth': 17,
    'p_reeval': 0.0, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': '1', 'pop_size': 250, 'generations': 500,
    'ri': 0.0, 'ei':0.0,
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'Spambase1', 'prob_type': prob_type, 
    'function_set': function_set,'function_mat': function_mat, 'terminal_set': terminal_set, 'terminal_mat': terminal_mat, 'data_types': data_types,
    'terminal_params': terminal_params, 'parent_selection': parent_selection}    

    for prob in problems:
        configs['problem'] = prob
        configs['domain'] = None
        configs['ndimensions'] = 4
        configs['npoints'] = 25
        
        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    for ri, ei in immigrants:
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut
                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs,suffix)

if __name__ == '__main__':
    for parent_selection in ["gp.tournament_selection"]:
        for arithmetic, consts in [(False, False), (False, True), (True, False), (True, True), (True, True)]:
            make_configs_sr_mat(arithmetic, consts, parent_selection)
            make_configs_NMultiplexer_mat(parent_selection)
            make_configs_sf_mat(parent_selection)
            make_config_Spambase_mat(parent_selection)
            make_config_LunarLander(parent_selection)