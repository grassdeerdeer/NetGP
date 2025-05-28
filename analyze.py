
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))









#==========================
import argparse
from glob import glob
from yaml import load, Loader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import importlib
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
import time
import json
# internal






def jsonify(d):
    """recursively formats dicts for json serialization"""
    if isinstance(d, list):
        d_new = []
        for v in d:
            d_new.append(jsonify(v))
        return d_new
    elif isinstance(d, dict):
        for k in d.keys():
            d[k] = jsonify(d[k])
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif d.__class__.__name__.startswith('int'):
        return int(d)
    elif d.__class__.__name__.startswith('float'):
        return float(d)
    elif isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.values.tolist()
    elif isinstance(d, bool):
        return d
    elif d == None:
        return None
    elif not isinstance(d, str):
        print("WARNING: attempting to store ",d,"as a str for json")
        return str(d)
    return d


def read_file(filename, label='target', use_dataframe=True, sep=None):
    
    if filename.endswith('gz'):
        compression = 'gzip'
    else:
        compression = None
    
    print('compression:',compression)
    print('filename:',filename)

    input_data = pd.read_csv(filename, sep=sep, compression=compression)
     
    # clean up column names
    clean_names = {k:k.strip().replace('.','_') for k in input_data.columns}
    input_data = input_data.rename(columns=clean_names)

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1)
    if not use_dataframe:
        X = X.values
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names



def evaluate_model(dataset, 
    results_path,
    save_file,
    metadata,
    random_state,
    est_name,
    op=None,
    test=False,
    use_dataframe=True):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')
    np.random.seed(random_state)
    features, labels, feature_names =  read_file(
        dataset, 
        use_dataframe=use_dataframe
    )

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)
    X_test,X_train = X_test.values.T, X_train.values.T
    
    

    # import algorithm 
    print('import '+ est_name)
    algorithm = importlib.import_module(est_name)
    eva_module = importlib.import_module("GenDataset."+metadata['problem_name'])
    if hasattr(eva_module, metadata['eva']):
        eva_function = getattr(eva_module, metadata['eva'])


    if hasattr(algorithm, "run"):
        time_time = time.time()
        best_prog = algorithm.run(X_train, y_train, feature_names, op, eva_function)
        time_time = time.time() - time_time
        results = {
        'dataset': dataset,
        'algorithm':est_name,
        'random_state':random_state,
        'time_time': time_time, 
        'op': op,
        'symbolic_latex': best_prog.get_infix_pretty(do_simplify=True),
        'symbolic_sympy': best_prog.get_infix_sympy(do_simplify=True),
        'symbolic_str': best_prog.get_infix_str(),
    }
        ##################################################
        # scores
        ##################################################
        for fold, target, X in  [ 
                                ['train', y_train, torch.tensor(X_train)], 
                                ['test', y_test, torch.tensor(X_test)]
                                ]:
            
            y_pred = np.asarray(best_prog.execute(X).detach().cpu().numpy()).reshape(-1,1)
            print('y_pred:',y_pred.shape)
            for score, scorer in [('mse',mean_squared_error),
                                ('pde',eva_function),
                                ('r2', r2_score),
                                ]:
                if score == 'r2':
                    # 计算R2分数
                    results[score + '_' + fold] = scorer(target, y_pred, force_finite=True)
                elif score == 'pde':
                    # X = torch.tensor(X).tolist()  
                    # for i in range(len(X)):
                    #     X[i]=torch.tensor(X[i],requires_grad=True)
                    
                    y_pred, X_temp = best_prog.torch_exec(X, best_prog.tokens, best_prog.free_const_values)
                    
                    results[score + '_' + fold] = scorer(y_pred[0],X_temp).detach().cpu().numpy()
                    y_pred = np.asarray(best_prog.execute(X).detach().cpu().numpy()).reshape(-1,1)
                else:
                    results[score + '_' + fold] = scorer(target, y_pred) 
        
        ##################################################
        # write to file
        ##################################################

        

        save_file = os.path.join(
            results_path,
            '_'.join([ml, str(random_state)])
        )

        print('save_file:',save_file)

        with open(save_file + '.json', 'w') as out:
            json.dump(jsonify(results), out, indent=4)

    print('evaluate')
    pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description="An analyst for solving PDE.", add_help=False)
    parser.add_argument('--DATASET_DIR', type=str,nargs='?', default='PDEdataset/**/*', help='Dataset directory like (PDEdataset)')
    parser.add_argument('-results',action='store',dest='RDIR',default='resultsv2',
            type=str,help='Results directory')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default="netgp",
            type=str, help='Comma-separated list of ML methods to use (should '
            'correspond to a py file name in methods/)')
    parser.add_argument('-seed',action='store',dest='SEED',default=1,
            type=int, help='A specific random seed')
    parser.add_argument('-op',action='store',dest='OP',default=["mul", "add", "sub", "sin", "exp"],type=list)
    args = parser.parse_args()
    ml = args.LEARNERS
    ops = args.OP
    # args.DATASET_DIR = "./PDEdataset/Advection/1_1D/"

    #datasets= ["NewGPSR/PDEdataset/Heat/1-3D/data.tsv.gz"]
    if args.DATASET_DIR.endswith('/'):
        datasets = [args.DATASET_DIR+'data.tsv.gz']
    elif args.DATASET_DIR.endswith('*'):
        print('capturing glob',args.DATASET_DIR+'/*.tsv.gz')
        datasets = glob(args.DATASET_DIR+'/*.tsv.gz')
    else:
        datasets = glob(args.DATASET_DIR+'/*/*.tsv.gz')
    
    print('found',len(datasets),'datasets')

    random_state = args.SEED
    for dataset in datasets:
        dataset = dataset.replace('\\', '/')
        metadata = load(
                open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
                    Loader=Loader)
        
        dataname = dataset.split('/')[-1].split('.tsv.gz')[0]
        results_path = os.path.join(args.RDIR, dataset.split(dataname+'.tsv.gz')[0])
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        save_file = os.path.join(
            results_path,
            '_'.join([ml, str(random_state)+".json"])
        )
        if (os.path.exists(save_file) ):
            print('file exists, skipping',save_file)
            continue

        evaluate_model(dataset=dataset, 
        results_path=results_path,
        save_file=save_file,
        metadata=metadata,
        random_state=random_state,
        est_name=ml,
        op=ops,
        test=False,)
        
        