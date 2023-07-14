import os
import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml
from sklearn import metrics
from app.algo import Coordinator, Client


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.models = {}

        self.train_filename = None
        self.test_filename = None

        self.pred_output = None
        self.proba_output = None
        self.test_output = None
        self.sep = ","
        self.label_column = None
        self.mode = None
        self.dir = "."
        self.splits = {}
        self.test_splits = {}
        self.betas = {}
        self.iter_counter = 0
        self.betas_finished = {}
        self.max_iter = None

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_logistic_regression']
            self.train_filename = config['input']['train']
            self.test_filename = config['input']['test']

            self.pred_output = config['output']['pred']
            self.proba_output = config['output']['proba']
            self.test_output = config['output']['test']

            self.sep = config['format']['sep']
            self.label_column = config['format']['label']

            self.mode = config['split']['mode']
            self.dir = config['split']['dir']

            self.max_iter = config['algo']['max_iterations']

        if self.mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
            self.test_splits = dict.fromkeys(self.splits.keys())
            self.models = dict.fromkeys(self.splits.keys())
            self.betas = dict.fromkeys(self.splits.keys())
            self.betas_finished = dict.fromkeys(self.splits.keys())
        else:
            self.splits[self.INPUT_DIR] = None
            self.test_splits[self.INPUT_DIR] = None
            self.models[self.INPUT_DIR] = None
            self.betas[self.INPUT_DIR] = None
            self.betas_finished[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocessing = 3
        state_local_computation = 4
        state_wait_for_aggregation = 5
        state_global_aggregation = 6
        state_writing_results = 7
        state_finishing = 8

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("Initializing")
                if self.id is not None:  # Test if setup has happened already
                    print("Coordinator", self.coordinator)

                    state = state_read_input

            if state == state_read_input:
                print('Read input and config')
                self.read_config()

                for split in self.splits.keys():
                    print(f'Read {split}')
                    if self.coordinator:
                        self.models[split] = Coordinator()
                    else:
                        self.models[split] = Client()
                    train_path = split + "/" + self.train_filename
                    test_path = split + "/" + self.test_filename
                    X = pd.read_csv(train_path, sep=self.sep, dtype="float32")
                    y = X.loc[:, self.label_column]
                    X = X.drop(self.label_column, axis=1)
                    X_test = pd.read_csv(test_path, sep=self.sep, dtype="float32")
                    y_test = X_test.loc[:, self.label_column]

                    y_test.to_csv(split.replace("/input", "/output") + "/" + self.test_output, index=False)
                    X_test = X_test.drop(self.label_column, axis=1)

                    self.splits[split] = [X, y]
                    self.test_splits[split] = [X_test, y_test]

                state = state_preprocessing

            if state == state_preprocessing:
                for split in self.splits.keys():
                    print(f'Preprocess {split}')
                    model = self.models[split]
                    X, y, beta = model.init(self.splits[split][0], self.splits[split][1])
                    self.splits[split] = [X, y]
                    self.models[split] = model
                    self.betas[split] = beta
                state = state_local_computation

            if state == state_local_computation:
                print("Perform local beta update")
                self.progress = 'local beta update'
                self.iter_counter = self.iter_counter + 1
                print(f'Iteration {self.iter_counter}')
                data_to_send = {}
                for split in self.splits.keys():
                    print(f'Compute {split}')
                    try:
                        data_to_send[split] = self.models[split].compute_derivatives(self.splits[split][0],
                                                                                     self.splits[split][1],
                                                                                     self.betas[split])
                    except FloatingPointError:
                        data_to_send[split] = "early_stop"

                data_to_send = jsonpickle.encode(data_to_send)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_aggregation:
                print("Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("Received aggregation data from coordinator.")
                    data_decoded = jsonpickle.decode(self.data_incoming[0])
                    self.betas, self.betas_finished = data_decoded[0], data_decoded[1]
                    self.data_incoming = []
                    if False in self.betas_finished.values() and self.max_iter > self.iter_counter:
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        for split in self.splits:
                            self.models[split].set_coefs(self.betas[split])
                        state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    print("Received data of all clients")
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    for split in self.splits.keys():
                        if not self.betas_finished[split]:
                            print(f'Aggregate {split}')
                            split_data = []
                            for client in data:
                                split_data.append(client[split])
                            beta, beta_finished = self.models[split].aggregate_beta(split_data)
                            self.betas[split] = beta
                            self.betas_finished[split] = beta_finished
                    data_to_broadcast = jsonpickle.encode([self.betas, self.betas_finished])
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    print(f'[COORDINATOR] Broadcasting computation data to clients', flush=True)
                    if False in self.betas_finished.values() and self.max_iter > self.iter_counter:
                        print(f'Beta update not finished for all splits.')
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        for split in self.splits.keys():
                            self.models[split].set_coefs(self.betas[split])
                        state = state_writing_results

            if state == state_writing_results:
                print("Writing results")
                for split in self.splits.keys():
                    print(f'Write {split}')
                    model = self.models[split]

                    joblib.dump(model, split.replace("/input", "/output") + '/model.pkl')

                    y_pred = pd.DataFrame(model.predict(self.test_splits[split][0]), columns=["y_pred"])
                    # this part logs the client wise test accuracy and auroc
                    y_real = self.test_splits[split][1]
                    y_pred_val = model.predict(self.test_splits[split][0])
                    print("test Accuracy:", model.score(self.test_splits[split][0], self.test_splits[split][1]) )
                    print ("test AUROC:", metrics.roc_auc_score(y_real,y_pred_val))
                    y_proba = pd.DataFrame(model.predict_proba(self.test_splits[split][0]))
                    y_pred.to_csv(split.replace("/input", "/output") + "/" + self.pred_output, index=False)
                    y_proba.to_csv(split.replace("/input", "/output") + "/" + self.proba_output, index=False)

                if self.coordinator:
                    self.data_incoming.append('DONE')
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("Finishing")
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            time.sleep(1)


logic = AppLogic()
