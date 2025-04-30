# load modules and dataset
from ray.tune.progress_reporter import CLIReporter
from DP_server import *
from DP_load_dataset import *
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd

def run_dp(method, model, dataset, prn = True, seed = 123, trial = False, select_round = False,pref_bs=64,local_times=10,n_obj=2,
           weight_merged=None, adaptive_dist=None, **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':

        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    # set up the server
    if method =='fedpf':
        head=Hypernet(n_obj=n_obj,num_classes=2, seed=seed)
        base = PF_MLP(num_features=num_features, num_classes=2, seed=seed)
        server = Server(BaseHeadSplit(base, head), info, train_prn=False, seed=seed, Z=Z,
                        ret=True, prn=prn, trial=trial, select_round=select_round, dataset=dataset, pref_bs=pref_bs, local_times=local_times)
    elif method =='fedgl':
        head=Hypernet(n_obj=n_obj,num_classes=2, seed=seed)
        base = PF_MLP(num_features=num_features, num_classes=2, seed=seed)
        server = Server(BaseHeadSplit(base, head), info, train_prn=False, seed=seed, Z=Z, adaptive_dist=adaptive_dist,
                        ret=True, prn=prn, trial=trial, select_round=select_round, dataset=dataset, pref_bs=pref_bs, local_times=local_times, weight_merged=weight_merged)
    else:
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed,
                        Z = Z, ret = True, prn = prn, trial = trial, select_round = select_round)
    # execute
    if method == 'fedavg':
        result = server.FedAvg(**kwargs)
    elif method == 'uflfb':
        result = server.UFLFB(**kwargs)
    elif method == 'fedfb':
        result = server.FedFB(**kwargs)
    elif method == 'fedpf':
        result = server.FedPF(**kwargs)
    elif method == 'fedgl':
        result = server.FedGL(**kwargs)
    elif method == 'cflfb':
        acc, dpdisp, classifier = server.CFLFB(**kwargs)
    elif method == 'fflfb':
        result = server.FFLFB(**kwargs)
    elif method == 'fairfed':
        result = server.FairFed(**kwargs)
    elif method == 'agnosticfair':
        result = server.FAFL(**kwargs)
    else:
        Warning('Does not support this method!')
        exit(1)


    return result

def sim_dp(method, model, dataset, num_sim = 5, seed = 0, resources_per_trial = {'cpu':4}, **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':
        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    if method == 'fedavg':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .002, .005, .01, .02])}
        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'loss',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("loss", "min", "last")
        learning_rate = best_trial.config['lr']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'fedfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {
                'alpha': tune.grid_search([.001, .05, .08, .1, .2, .5, 1, 2])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha = params['lr'], params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'cflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005]),
                'alpha': tune.grid_search([.001, .05, .08, .1, .2]),
                'rounds': tune.grid_search([1,10])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], outer_rounds = config['rounds'], inner_epochs = 300//config['rounds'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 50)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha, rounds = params['lr'], params['alpha'], params['rounds']
        print("The hyperparameter we select is | learning rate: %.4f | alpha: %.4f " % (learning_rate, alpha))

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, outer_rounds = rounds, inner_epochs = 300//rounds, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'uflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        num_clients = len(info[2])
        if num_clients <= 2:
            params_array = cartesian([[.001, .01, .1]]*num_clients).tolist()
            # params_array = cartesian([[.01]]*num_clients).tolist()
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = config['alpha'], **kwargs)
        else:
            params_array = [.001, .002, .005, .01, .02, .05, .1, 1]
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = [config['alpha']] * num_clients, **kwargs)
        config = {'alpha': tune.grid_search(params_array)}

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1)

        params = analysis.get_best_config(metric = "DP Disp", mode = "min")
        alpha = params['alpha']
        df = analysis.results_df[['accuracy', 'DP Disp']]

        print('--------------------------------Start Simulations--------------------------------')
        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            if num_clients <= 2:
                result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = alpha, **kwargs)
            else:
                result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = [alpha] * num_clients, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std
    
    elif method == 'fflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        num_clients = len(info[2])
        if num_clients <= 2:
            params_array = cartesian([[.001, .01, .1]]*num_clients).tolist()
            # params_array = cartesian([[.01]]*num_clients).tolist()
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = 0.005, alpha = config['alpha'], **kwargs)
        else:
            params_array = [.001, .002, .005, .01, .02, .05, .1, 1]
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = 0.005, alpha = [config['alpha']] * num_clients, **kwargs)
        config = {'alpha': tune.grid_search(params_array)}

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        alpha = params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            if num_clients <= 2:
                result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = 0.005, alpha = alpha, **kwargs)
            else:
                result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = 0.005, alpha = [alpha] * num_clients, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'fairfed':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005,]),
                'alpha': tune.grid_search([.0001, .001, .01]),
                'beta': tune.grid_search([0.02, 1, 50])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], beta = config['beta'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha, beta = params['lr'], params['alpha'], params['beta']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, beta = beta, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'agnosticfair':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005,]),
                'penalty': tune.grid_search([0, 10, 100, 200, 500, 1000, 5000, 10000])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], penalty = config['penalty'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, penalty = params['lr'], params['penalty']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, penalty = penalty, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std
    else: 
        Warning('Does not support this method!')
        exit(1)

def sim_dp_man(method, model, dataset, num_sim = 5, seed = 0, select_round = False, pref_bs=64, local_times=10,
               weight_merged=None, adaptive_dist=None,  **kwargs):
    global_results = []
    local_results = []
    global_results_adaptmerge = []
    pref_collect = []
    Val_HV = []
    for seed in range(num_sim):
        test_re, test_global, collected_pref, val_hv = run_dp(method, model, dataset, prn = True, seed = seed, trial = False, adaptive_dist=adaptive_dist,
                              select_round = select_round,pref_bs=pref_bs, local_times=local_times,weight_merged=weight_merged, **kwargs)
        if method == 'fedgl':
            local_results.append(test_global[0])
            global_results.append(test_global[1])
            global_results_adaptmerge.append(test_global[2])
        else:
            local_results.append(test_re)
            global_results.append(test_global)
        pref_collect.append(collected_pref)
        Val_HV.append(val_hv)
    # np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}.npy', np.array(results))
    heter = ''
    cn = 10
    np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_{cn}_local.npy', np.array(local_results))
    if method == 'fedgl':
        np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_nonmerge_{cn}_global.npy',
                np.array(global_results))
        np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_withmerge_{cn}_global.npy',
                np.array(global_results_adaptmerge))
    else:
        np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_{cn}_global.npy', np.array(global_results))
    np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_{cn}_pref.npy', np.array(pref_collect))
    np.save(f'result/{method}_{dataset}_{seed}_{pref_bs}_{local_times}_dist_{adaptive_dist}{heter}_{cn}_valhv.npy', np.array(Val_HV))
    # df = pd.DataFrame(results)
    # acc_mean, dp_mean = df.mean()
    # acc_std, dp_std = df.std()
    # print("Result across %d simulations: " % num_sim)
    # print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
    # return acc_mean, acc_std, dp_mean, dp_std


if __name__ == "__main__":
    import sys, os
    import argparse
    working_dir = '.'
    sys.path.insert(1, os.path.join("/home/yerongguang/Fairness/FedFB"))
    os.environ["PYTHONPATH"] = os.path.join("/home/yerongguang/Fairness/FedFB")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='fedavg', type=str)
    parser.add_argument('--model', default='multilayer perceptron', type=str)
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--num_sim', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_rounds', default=15, type=int)
    parser.add_argument('--pref_bs', default=8, type=int)
    parser.add_argument('--local_times', default=15, type=int)
    parser.add_argument('--n_obj', default=2, type=int)
    parser.add_argument('--weight_merged', default=None)
    parser.add_argument('--adaptive_dist', default=None)
    args = parser.parse_args()
    print(args)
    if args.name in ['uflfb', 'cflfb', 'fflfb', 'agnosticfair']:
        sim_dp_man(
            method=args.name,
            model=args.model,
            dataset=args.dataset,
            num_sim=args.num_sim,
            seed=args.seed,

        )
    else:
        sim_dp_man(
            method=args.name,
            model=args.model,
            dataset=args.dataset,
            num_sim=args.num_sim,
            seed=args.seed,
            num_rounds=args.num_rounds,
            pref_bs = args.pref_bs,
            local_times = args.local_times,
            n_obj=args.n_obj,
            weight_merged = args.weight_merged,
            adaptive_dist =args.adaptive_dist
        )
    # sim_dp(
    #     method=args.name,
    #     model=args.model,
    #     dataset=args.dataset,
    #     seed=args.seed,
    #     num_rounds=args.num_rounds,
    #     resources_per_trial={'cpu': 4}
    # )

