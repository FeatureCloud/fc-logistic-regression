import os
import shutil
import threading
import time

import numpy as np
import joblib
import jsonpickle
import pandas as pd
import yaml

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State
from algo import Coordinator, Client

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """
    
    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        print("Initializing")
        if self.id is not None:  # Test if setup has happened already
            print("Coordinator", self.is_coordinator)
        return 'read input'


@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """
    
    def register(self):
        self.register_transition('preprocessing', Role.BOTH)
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            print('Read input and config')
            self.read_config()
        
            splits = self.load('splits')
            test_splits = self.load('test_splits')
            models = self.load('models')

            for split in splits.keys():
                print(f'Read {split}')
                if self.is_coordinator:
                    models[split] = Coordinator()
                else:
                    models[split] = Client()
                train_path = split + "/" + self.load('train_filename')
                test_path = split + "/" + self.load('test_filename')
                X = pd.read_csv(train_path, sep=self.load('sep'), dtype="float32")
                y = X.loc[:, self.load('label_column')]
                X = X.drop(self.load('label_column'), axis=1)
                X_test = pd.read_csv(test_path, sep=self.load('sep'), dtype="float32")
                y_test = X_test.loc[:, self.load('label_column')]
                
                y_test.to_csv(split.replace("/input", "/output") + "/" + self.load('test_output'), index=False)
                X_test = X_test.drop(self.load('label_column'), axis=1)

                splits[split] = [X, y]
                test_splits[split] = [X_test, y_test]

            return 'preprocessing'
            
        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return 'read input'
        
    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        models = {}
        splits = {}
        test_splits = {}
        betas = {}
        betas_finished = {}
        
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_logistic_regression']
                
            self.store('train_filename', config['input']['train'])
            self.store('test_filename', config['input']['test'])

            self.store('pred_output', config['output']['pred'])
            self.store('proba_output', config['output']['proba'])
            self.store('test_output', config['output']['test'])

            self.store('sep', config['format']['sep'])
            self.store('label_column', config['format']['label'])

            self.store('mode', config['split']['mode'])
            self.store('dir', config['split']['dir'])

            if self.is_coordinator:
                self.store('max_iter', config['algo']['max_iterations'])
                self.broadcast_data(self.load('max_iter'), send_to_self=False)

            else:
                max_iter = self.await_data()
                self.store('max_iter', max_iter)
                
        if self.load('mode') == "directory":
            splits = dict.fromkeys([f.path for f in os.scandir(f"{self.load('INPUT_DIR')}/{self.load('dir')}") if f.is_dir()])
            test_splits = dict.fromkeys(splits.keys())
            models = dict.fromkeys(splits.keys())
            betas = dict.fromkeys(splits.keys())
            betas_finished = dict.fromkeys(splits.keys())
        else:
            splits[self.load('INPUT_DIR')] = None
            test_splits[self.load('INPUT_DIR')] = None
            models[self.load('INPUT_DIR')] = None
            betas[self.load('INPUT_DIR')] = None
            betas_finished[self.load('INPUT_DIR')] = None
           
        for split in splits.keys():
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
            shutil.copyfile(self.load('INPUT_DIR') + '/config.yml', self.load('OUTPUT_DIR') + '/config.yml')
        print(f'Read config file.', flush=True)
        
        self.store('models', models)
        self.store('splits', splits)
        self.store('test_splits', test_splits)
        self.store('betas', betas)
        self.store('betas_finished', betas_finished)
            
            
@app_state('preprocessing', Role.BOTH)
class PreprocessingState(AppState):
    """
    Preprocess model parameters.
    """
    
    def register(self):
        self.register_transition('local computation', Role.BOTH)
        self.register_transition('preprocessing', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            splits = self.load('splits')
            models = self.load('models')
            betas = self.load('betas')
            for split in splits.keys():
                print(f'Preprocess {split}')
                model = models[split]
                X, y, beta = model.init(splits[split][0], splits[split][1])
                splits[split] = [X, y]
                models[split] = model
                betas[split] = beta
            self.store('iteration', 0)
            return 'local computation'
        
        except Exception as e:
            self.log('error preprocessing', LogLevel.ERROR)
            self.update(message='error preprocessing', state=State.ERROR)
            print(e)
            return 'preprocessing'


@app_state('local computation', Role.BOTH)
class LocalComputationState(AppState):
    """
    Perform local computation and send the computation data to the coordinator.
    """
    
    def register(self):
        self.register_transition('global aggregation', Role.COORDINATOR)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        self.register_transition('local computation', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            print("Perform local beta update")
            self.store('iteration', self.load('iteration') + 1)
            print(f"Iteration {self.load('iteration')}")
            splits = self.load('splits')
            models = self.load('models')
            betas = self.load('betas')
            data_to_send = {}
            for split in splits.keys():
                print(f'Compute {split}')
                try:
                    data_to_send[split] = models[split].compute_derivatives(splits[split][0], splits[split][1],
                    betas[split])
                except FloatingPointError:
                    derivatives = "early_stop"
            data_to_send = jsonpickle.encode(data_to_send)
            self.send_data_to_coordinator(data_to_send)

            if self.is_coordinator:
                return 'global aggregation'
            else:
                print(f'[CLIENT] Sending computation data to coordinator', flush=True)
                return 'wait for aggregation'
                
        except Exception as e:
            self.log('error local computation', LogLevel.ERROR)
            self.update(message='error local computation', state=State.ERROR)
            print(e)
            return 'local computation'


@app_state('wait for aggregation', Role.PARTICIPANT)
class WaitForAggregationState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """
    
    def register(self):
        self.register_transition('local computation', Role.PARTICIPANT)
        self.register_transition('writing results', Role.PARTICIPANT)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            print("Wait for aggregation")
            data = self.await_data()
            print("Received aggregation data from coordinator.")
            data_decoded = jsonpickle.decode(data)
 
            betas, betas_finished = data_decoded[0], data_decoded[1]
            self.store('betas', betas)
            self.store('betas_finished', betas_finished)
            
            if False in betas_finished.values() and self.load('max_iter') > self.load('iteration'):
                return 'local computation'

            else:
                print("Beta update finished.")
                models = self.load('models')
                betas = self.load('betas')
                for split in self.load('splits'):
                    models[split].set_coefs(betas[split])
                return 'writing results'
        
        except Exception as e:
            self.log('error wait for aggregation', LogLevel.ERROR)
            self.update(message='error wait for aggregation', state=State.ERROR)
            print(e)
            return 'wait for aggregation'
       
       
@app_state('global aggregation', Role.COORDINATOR)
class GlobalAggregationState(AppState):
    """
    The coordinator receives the local computation data from each client and aggregates the weights.
    The coordinator broadcasts the global computation data to the clients.
    """

    def register(self):
        self.register_transition('local computation', Role.COORDINATOR)
        self.register_transition('writing results', Role.COORDINATOR)
        self.register_transition('global aggregation', Role.COORDINATOR)
    
    def run(self) -> str or None:
        try:
            print("Global computation")
            data = self.gather_data()
            print("Received data of all clients")
            data = [jsonpickle.decode(client_data) for client_data in data]
          
            models = self.load('models')
            betas = self.load('betas')
            betas_finished = self.load('betas_finished')
            for split in self.load('splits'):
                if not betas_finished[split]:
                    print(f'Aggregate {split}')
                    split_data = []
                    for client in data:
                        split_data.append(client[split])
                    beta, beta_finished = models[split].aggregate_beta(split_data)
                    betas[split] = beta
                    betas_finished[split] = beta_finished

            data_to_broadcast = jsonpickle.encode([betas, betas_finished])
            self.broadcast_data(data_to_broadcast, send_to_self=False)
            print(f'[COORDINATOR] Broadcasting computation data to clients', flush=True)
 
            if False in self.load('betas_finished').values() and self.load('max_iter') > self.load('iteration'):
                print(f'Beta update not finished for all splits.')
                return 'local computation'
            else:
                print("Beta update finished.")
                betas = self.load('betas')
                for split in self.load('splits'):
                    models[split].set_coefs(betas[split])
                return 'writing results'
                
        except Exception as e:
            self.log('error global aggregation', LogLevel.ERROR)
            self.update(message='error global aggregation', state=State.ERROR)
            print(e)
            return 'global aggregation'
    
    
@app_state('writing results', Role.BOTH)
class WritingResultsState(AppState):
    """
    Writes the results of the global computation.
    """
    
    def register(self):
        self.register_transition('terminal', Role.BOTH)
        self.register_transition('writing results', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            print("Writing results")
            models = self.load('models')
            test_splits = self.load('test_splits')
            
            for split in self.load('splits').keys():
                print(f'Write {split}')
                model = models[split]

                joblib.dump(model, split.replace("/input", "/output") + '/model.pkl')

                y_pred = pd.DataFrame(model.predict(test_splits[split][0]), columns=["y_pred"])
                y_proba = pd.DataFrame(model.predict_proba(test_splits[split][0]))
                y_pred.to_csv(split.replace("/input", "/output") + "/" + self.load('pred_output'), index=False)
                y_proba.to_csv(split.replace("/input", "/output") + "/" + self.load('proba_output'), index=False)

            self.send_data_to_coordinator('DONE')
        
            if self.is_coordinator:
                print("Finishing")
                self.gather_data()
        
            return 'terminal'
            
        except Exception as e:
            self.log('error writing results', LogLevel.ERROR)
            self.update(message='error writing results', state=State.ERROR)
            print(e)
            return 'writing results'
