from .train_agent import Agent, get_scaling, scale_data, BPNN
from .fp_calculator import set_sym, db_to_fp
import torch
from ase.db import connect
import os
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import numpy as np
from time import sleep
import copy


class Ensemble_Trainer():
    def __init__(self, train_db, model_path, raw_fp_params, ensemble_size, nn_params, torch_client):
        self.model_path = model_path
        self.train_db = train_db
        self.ensemble_size = ensemble_size
        self.raw_fp_params = raw_fp_params
        self.nn_params = nn_params
        self.torch_client = torch_client
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        self.train_data = None
        self.valid_data = None
    

    def calculate_fp(self):
        el, Gs, cutoff = self.raw_fp_params['el'], self.raw_fp_params['gs'], self.raw_fp_params['cutoff']
        g2_etas, g2_Rses = self.raw_fp_params['g2_etas'], self.raw_fp_params['g2_Rses']
        g4_etas, g4_zetas = self.raw_fp_params['g4_etas'], self.raw_fp_params['g4_zetas']
        g4_lambdas = self.raw_fp_params['g4_lambdas']
        params_set = set_sym(el, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses, 
                            g4_etas=g4_etas, g4_zetas=g4_zetas, g4_lambdas=g4_lambdas)
        train_data = db_to_fp(self.train_db, params_set)
        # torch.save(train_data, os.path.join(self.model_path, 'train_set_data.sav'))
        # scale_file = os.path.join(self.model_path, 'train_set_scale.sav')
        scale = get_scaling(train_data, add_const=1e-10)
        # torch.save(scale, scale_file)
        train_data = scale_data(train_data, scale)
        valid_data = copy.deepcopy(train_data)
        # self.train_data = train_data
        # self.valid_data = valid_data
        return train_data, valid_data

    def train_ensemble(self):
        if self.torch_client is None:  # train models sequentially 
            for m in range(self.ensemble_size):
                self.train_nn(m)
        else:  # train models parallelly using dask, this does not work, Exception: can't pickle _cffi_backend.__CDataOwn objects       
            ids = list(np.arange(self.ensemble_size))
            L = self.torch_client.map(self.train_nn_dask, ids)
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
        train_data, valid_data = self.calculate_fp()
        agent = Agent(train_data=train_data, valid_data=valid_data, model_path=model_path,
                    layer_nodes=layer_nodes, activation=activations, lr=lr, scale_const=1.0)
        agent.train(log_name=log_name, n_epoch=3000, interupt=True, val_interval=20, is_force=True, 
                    nrg_convg=2, force_convg=7, max_frs_convg=50, nrg_coef=1, force_coef=1)
        return
    
    def train_nn_dask(self, m):
        train_data, valid_data = self.calculate_fp()
        lr = self.nn_params['lr']
        model_path = f'model-{m}.sav'
        log_name = f'log-{m}.txt'
        layer_nodes = self.nn_params['layer_nodes']
        activations = self.nn_params['activations']
        agent = Agent(train_data=train_data, valid_data=valid_data, model_path=model_path,
                    layer_nodes=layer_nodes, activation=activations, lr=lr, scale_const=1.0)
        agent.train(log_name=log_name, n_epoch=3000, interupt=True, val_interval=20, is_force=True, 
                    nrg_convg=2, force_convg=7, max_frs_convg=50, nrg_coef=1, force_coef=1)
        return agent.model


