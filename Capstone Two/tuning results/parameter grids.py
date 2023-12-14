'''This is not an real python script, it just serves as a record for hyperparameter tuning grids. Could format to JSON'''

tuning_grids = {
    'SGDRegressor': { 'iterations': 300, 'grid': {
        # leave loss function, penalty, learning_rate, fit_intercept at defaults: 'squared_error', 'L2', 'invscaling', True
        # vary alpha, tol(erance), eta0, power_t
        'alpha': loguniform(8e-5, 1e-2, 0, 1), # def=1e-4, float | higher values increase regularization strength
        'tol': loguniform(1e-5, 1e-2, 0, 1), # def=1e-3, float or None | convergence parameter
        'eta0': loguniform(1e-2, 8e-1, 0, 1), # def=0.01, float | initial learning rate
        'power_t': powerlaw(1, 0, 1), # def=0.25, float | exponent for invscaling learning rate: eta = eta0/pow(t, power_t)
    }},
    'Ridge': { 'iterations': 300, 'grid': {
        # vary alpha, tolerance/solver (see old sheet)
        'alpha': loguniform(0.01,10,0,1), # def=1, controls regularization strength [0,inf)
        'tol': loguniform(1e-2,0.95,0,1), # def=1e-4, specifies convergence critieria for solvers: sparse_cg, lsqr, sag, saga, lbfgs
        'solver':['saga'], # def='auto', previous grid search found good performance for saga, limit this search
    }},
    'KernelRidge': { 'iterations': 30, 'grid': {
        # vary alpha. if kernel is varied to something other than linear, then can also vary gamma
        'alpha': uniform(0,2), # def=1, controls regularization strength, must be positive float
        'kernel': ['linear'], # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
    }},
    'KNeighborsRegressor': { 'iterations': 60, 'grid': {
        # vary n_neighbors, weights, algorithm (k-d vs ball tree), leaf_size (keep default), p [0,1]
        'n_neighbors': randint(2,9,0), # def=5, 
        'weights': ['uniform','distance'], # def='uniform', weight function used in prediction. uniform or distance based (inverse relation)
        'algorithm': ['ball_tree', 'kd_tree'], # def='auto', brute-force, kd_tree, ball_tree
        'leaf_size': nbinom(30,0.5,1),
        'p':[1,2],
    }},
    # vary estimators, keep criterion at squared_error (friedman can work well), vary min_samples_split
    'ExtraTreesRegressor': {'iterations': 90, 'grid': {
        'n_estimators':boltzmann(0.01,200,10),  
    }},
    # vary learning rate, n_estimators. keep loss function as huber
    'GradientBoostingRegressor': {'iterations': 60, 'grid': {
        'loss':['huber'],
        'learning_rate': loguniform(.04,1,0,1),
        'n_estimators': boltzmann(0.01,200,10),
    }},
    'RandomForestRegressor': {'iterations': 60, 'grid': {
        'n_estimators':boltzmann(0.01,200,10),  
        'max_features':binom(14,0.5,0),
    }},
    'LGBMRegressor': {'iterations': 30, 'grid': {
        'num_leaves': randint(25,36,0), # def=31
        'min_data_in_leaf': randint(15,25,0),  #def=20, not listed as parameter
        'learning_rate':loguniform(0.01,0.5,0,1), # def=0.1
        'n_estimators':randint(100,300,0), # def=100, not listed as a parameter | boltzmann(0.001,500,50)
    }},
    'CatBoostRegressor': {'iterations': 20, 'grid': {
        'l2_leaf_reg':uniform(0,4), # def=3
        'random_strength':uniform(0,3), # def=1
        'learning_rate':beta(2,20,0,1), # def=0.043
    }},
    # XGB - Linear Booster
    'XGBRegressor_gblinear': {'iterations': 60, 'grid': {
        'lambda':loguniform(.001,1,0,1), # def=0, L2 regularization
        'updater':['shotgun'], # def='shotgun', descent algorithm to fit linear model
        'feature_selector': ['cyclic'], # def='cyclic', feature selection and ordering method
    }},
    # XGB - Tree
    # complexity: max_depth, min_child_weight, gamma
    # randomness: subsample, colsample_bytree. reduce eta while increasing num_round
    'XGBRegressor_gbtree': {'iterations': 60, 'grid': {
        'eta':beta(2,4,0,1), # def=0.3, learning_rate. [0,1]
        'gamma': loguniform(0.001,1,0,1), # def=0, minimum loss reduction, larger=more conservative. [0,inf)
        'max_depth': randint(2,15,0), # def=6, higher increases chances to overfit, use with learning rate
        'subsample': uniform(0.6,0.4), # def=1, (0,1]. practically vary 0.5-1
        'colsample_bytree': uniform(0.6,0.4), # def=1, subsample ratio of columns used when constructing each tree 
    }},
}