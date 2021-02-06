from .train_agent import Agent, get_scaling, scale_data, BPNN
from .fp_calculator import set_sym, db_to_fp
import torch
from ase.db import connect
import os
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import numpy as np
from time import sleep


class Ensemble_Trainer():
    def __init__(self, train_db_path, model_path, fp_params, ensemble_size, nn_params, torch_client):
        self.model_path = model_path
        self.train_db_path = train_db_path
        self.ensemble_size = ensemble_size
        self.params_set = fp_params
        self.nn_params = nn_params
        self.torch_client = torch_client
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        self.train_data = None
        self.valid_data = None
    

    def calculate_fp(self):
        train_db = connect(self.train_db_path)
        train_data = db_to_fp(train_db, self.params_set)
        torch.save(train_data, os.path.join(self.model_path, 'train_set_data.sav'))
        scale_file = os.path.join(self.model_path, 'train_set_scale.sav')
        scale = get_scaling(train_data, add_const=1e-10)
        torch.save(scale, scale_file)
        return

    def train_nn_dask(self, m, train_data, valid_data, layer_nodes, activations):
        lr = self.nn_params['lr']
        model_path = f'model-{m}.sav'
        log_name = f'log-{m}.txt'
        agent = Agent(train_data=train_data, valid_data=valid_data, model_path=model_path,
                    layer_nodes=layer_nodes, activation=activations, lr=lr, scale_const=1.0)
        agent.train(log_name=log_name, n_epoch=3000, interupt=True, val_interval=20, is_force=True, 
                    nrg_convg=2, force_convg=7, max_frs_convg=50, nrg_coef=1, force_coef=1)
        return agent.model


    def train_ensemble(self):
        data_set_path = os.path.join(self.model_path, 'train_set_data.sav')
        scale_path = os.path.join(self.model_path, 'train_set_scale.sav')
        train_data = torch.load(data_set_path)
        valid_data = torch.load(data_set_path)
        scale = torch.load(scale_path)
        train_data = scale_data(train_data, scale)
        valid_data = scale_data(valid_data, scale)

        if self.torch_client is None:  # train models sequentially 
            for m in range(self.ensemble_size):
                self.train_nn(m)
        else:  # train models parallelly using dask            
            ids = list(np.arange(self.ensemble_size))
            args = [[m, train_data, valid_data, self.nn_params['layer_nodes'], self.nn_params['activations']] for m in ids]
            L = self.torch_client.map(self.train_nn_dask, *args)
            res_models = self.torch_client.gather(L)
            for i in range(self.ensemble_size):
                tmp_model_path = os.path.join(self.model_path, f'model-{i}.sav')
                torch.save(res_models[i].state_dict(), tmp_model_path)
        return


    def train_nn(self, m):
        # create model and train
        model_path = os.path.join(self.model_path, f'model-{m}.sav')
        log_name = os.path.join(self.model_path, f'log-{m}.txt')
        layer_nodes = self.nn_params['layer_nodes']
        activations = self.nn_params['activations']
        lr = self.nn_params['lr']

        agent = Agent(train_data=self.train_data, valid_data=self.valid_data, model_path=model_path,
                    layer_nodes=layer_nodes, activation=activations, lr=lr, scale_const=1.0)
        agent.train(log_name=log_name, n_epoch=3000, interupt=True, val_interval=20, is_force=True, 
                    nrg_convg=2, force_convg=7, max_frs_convg=50, nrg_coef=1, force_coef=1)
        return
    

    def add(self, x, y):
        return x + y


    def test(self):
        ids = [[1, 2], [3, 4]]
        tmpL = self.torch_client.map(self.add, *ids)
        tmpres = self.torch_client.gather(tmpL)
        return tmpres