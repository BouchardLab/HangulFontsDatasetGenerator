import os, pickle
from functools import partial

import numpy as np

import torch
from torch.nn import (Sequential, Linear, ReLU, Sigmoid, Tanh,
                      Conv2d, ConvTranspose2d, MaxPool2d, MaxUnpool2d,
                      MSELoss, CrossEntropyLoss,
                      BatchNorm2d, BatchNorm1d, ModuleList)
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter

from ignite.engine import (Engine, Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss

from .layers.dropout import DropoutRescale, Dropout2dRescale
from .layers.linear import LinearInfoLognormal
from .layers.convolutional import Conv2dResnetBlock
from .losses import diagonal_logdet
from .BVAE import BVAE

_activation = {'relu': ReLU,
               'tanh': Tanh,
               'sigmoid': Sigmoid}

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SequentialAllActivations(Sequential):
    """Torch Sequential Module with one extra function to return all
    layer activations.
    """
    def forward(self, input, return_all=False):
        if return_all:
            output = []
            for module in self._modules.values():
                input = module(input)
                output.append(input)
            return output
        else:
            return super(SequentialAllActivations, self).forward(input)


def create_summary_writer(model, data_loader, save_folder, model_id, device='cpu', conv=False):
    """Create a logger.

    Parameters
    ----------
    model
        Pytorch model.
    data_loader
        Pytorch DataLoader.
    save_folder: str
        Base location to save models and metadata.
    model_id: str
        Model/hp ID.

    Returns
    -------
    writer
        Logger object.
    """
    model.eval()
    writer = SummaryWriter(os.path.join(save_folder, model_id))
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x = x.to(device)
    if conv:
        x = x.unsqueeze(1)
    with writer:
        try:
            writer.add_graph(model, x)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
    return writer

def classifier_train_loop(ds, params, settings, save_folder, model_id, funcs,
                          valid_name, kind='dense', n_classes=None,
                          device='cpu', max_epochs=100):
    """Train, validate, and save a model.

    Parameters
    ----------
    ds
        Train, valid, and test data loaders.
    params: dict
        Dictionary of model-specific hyperparameters.
    settings: dict
        Dictionary of model-common hyperparameters.
    save_folder: str
        Base location to save models and metadata.
    model_id: str
        Model/hp ID.
    funcs
        Iterable of function metrics to score models on.
    valid_name : str
        Name of metric to validate on.
    kind : str
        Type of model.
    n_classes : int
        Number of classes for classification model.
    """
    ds_train, ds_valid, ds_test = ds
    if n_classes is None:
        y_shape = ds_train.dataset.tensors[1].shape[1]
    else:
        y_shape = n_classes

    args = (ds_train.dataset.tensors[0].unsqueeze(0).shape, y_shape, params, settings)
    print('Not params')
#     model = make_dense_model(*args).to(device)
    model = make_conv2d_model(*args).to(device)
    with create_summary_writer(model, ds_train, save_folder, model_id, device='cuda') as writer:
        with open(os.path.join(save_folder, model_id, 'model_params.pkl'), 'wb') as f:
            pickle.dump(args, f)
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        opt = torch.optim.SGD(model.parameters(),
                              lr=lr, momentum=mom, weight_decay=wd)
        sched = ReduceLROnPlateau(opt, factor=.5, patience=10)

        if settings.get('info', False):
            loss = partial(diagonal_logdet, model, funcs['loss']._loss_fn,
                           params['info_beta'])
            funcs['info_loss'] = Loss(loss)
        else:
            loss = funcs['loss']._loss_fn

        trainer = create_supervised_trainer(model, opt, loss,
                                            device=device)
        train_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)
        valid_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)
        test_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(ds_valid)
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, valid_avg_accuracy, avg_nll))
            writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("valdation/avg_accuracy", valid_avg_accuracy, engine.state.epoch)
            writer.add_scalar("valdation/avg_error", 1.-valid_avg_accuracy, engine.state.epoch)
            test_evaluator.run(ds_test)
            metrics = test_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            writer.add_scalar("test/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("test/avg_accuracy", avg_accuracy, engine.state.epoch)
            writer.add_scalar("test/avg_error", 1.-avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['accuracy']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch),
                            batch_size=params['batch_size'])
            train_evaluator.run(ds)
            metrics = train_evaluator.state.metrics
            accuracy = metrics['accuracy']
            nll = metrics['loss']
            iter = (engine.state.iteration - 1) % len(ds_train) + 1
            if (iter % 100) == 0:
                print("Epoch[{}] Iter[{}/{}] Accuracy: {:.2f} Loss: {:.2f}"
                      .format(engine.state.epoch, iter, len(ds_train), accuracy, nll))
            writer.add_scalar("batchtraining/detloss", nll, engine.state.epoch)
            writer.add_scalar("batchtraining/accuracy", accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/error", 1.-accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/loss", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_lr(engine):
            writer.add_scalar("lr", opt.param_groups[0]['lr'], engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(ds_train)
            metrics = train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
            writer.add_scalar("training/avg_error", 1.-avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_value(engine):
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics[valid_name]
            return valid_avg_accuracy
        checkpoint = ModelCheckpoint(os.path.join(save_folder, model_id), model_id,
                                     score_function=validation_value,
                                     score_name='valid_{}'.format(valid_name))
        early_stopping = EarlyStopping(20, score_function=validation_value,
                                       trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'model': model})
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        # kick everything off
        trainer.run(ds_train, max_epochs=max_epochs)


def regressor_train_loop(ds, params, settings, save_folder, model_id, funcs,
                         valid_name, kind='dense',
                         device='cpu', max_epochs=100, minimize_valid=True):
    """Train, validate, and save a model.

    Parameters
    ----------
    ds
        Train, valid, and test data loaders.
    params: dict
        Dictionary of model-specific hyperparameters.
    settings: dict
        Dictionary of model-common hyperparameters.
    save_folder: str
        Base location to save models and metadata.
    model_id: str
        Model/hp ID.
    funcs
        Iterable of function metrics to score models on.
    valid_name : str
        Name of metric to validate on.
    kind : str
        Type of model.
    n_classes : int
        Number of classes for classification model.
    """
    ds_train, ds_valid, ds_test = ds
    y_shape = ds_train.dataset.tensors[1].shape[1]

    if kind == 'dense':
        args = (ds_train.dataset.tensors[0].shape[1], y_shape, params, settings)
        model = make_dense_model(*args)
    elif kind == 'conv':
        args = (ds_train.dataset.tensors[0].shape[1:], y_shape, params, settings)
        model = make_conv2d_model(*args)
    else:
        raise ValueError
    if settings.get('parallel', False) and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    #model.to(device)

    with create_summary_writer(model, ds_train, save_folder, model_id) as writer:
        with open(os.path.join(save_folder, model_id, 'model_params.pkl'), 'wb') as f:
            pickle.dump(args, f)
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        opt = torch.optim.Adam(model.parameters())
        sched = ReduceLROnPlateau(opt, factor=.5, patience=10)

        loss = funcs['loss']._loss_fn

        trainer = create_supervised_trainer(model, opt, loss,
                                            device=device)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        train_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)
        valid_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)
        test_evaluator = create_supervised_evaluator(model, metrics=funcs, device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(ds_valid)
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['loss']
            print("Validation Results - Epoch: {}  Avg loss: {:.3f}"
                  .format(engine.state.epoch, avg_nll))
            writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
            test_evaluator.run(ds_test)
            metrics = test_evaluator.state.metrics
            avg_nll = metrics['loss']
            print("Test Results - Epoch: {}  Avg loss: {:.3f}"
                  .format(engine.state.epoch, avg_nll))
            writer.add_scalar("test/avg_loss", avg_nll, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['loss']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch),
                            batch_size=params['batch_size'])
            train_evaluator.run(ds)
            metrics = train_evaluator.state.metrics
            nll = metrics['loss']
            iter = (engine.state.iteration - 1) % len(ds_train) + 1
            if (iter % 100) == 0:
                print("Epoch[{}] Iter[{}/{}] Loss: {:.3f}"
                      .format(engine.state.epoch, iter, len(ds_train), nll))
            writer.add_scalar("batchtraining/loss", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_lr(engine):
            writer.add_scalar("lr", opt.param_groups[0]['lr'], engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(ds_train)
            metrics = train_evaluator.state.metrics
            avg_nll = metrics['loss']
            print("Training Results - Epoch: {}  Avg loss: {:.3f}"
                  .format(engine.state.epoch, avg_nll))
            writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_value(engine):
            metrics = valid_evaluator.state.metrics
            ll = metrics[valid_name]
            if not minimize_valid:
                ll = -ll
            return ll

        checkpoint = ModelCheckpoint(os.path.join(save_folder, model_id), model_id,
                                     score_function=validation_value,
                                     score_name='valid_{}'.format(valid_name))
        early_stopping = EarlyStopping(20, score_function=validation_value,
                                       trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'model': model})
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        # kick everything off
        trainer.run(ds_train, max_epochs=max_epochs)


def make_dense_model(input_dim, output_dim, params, settings):
    """Create a densely connected model given hyperparameters.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    output_dim: int
        Number of output features/classes.
    params: dict
        Dictionary of model-specific hyperparameters.
    settings: dict
        Dictionary of model-common hyperparameters.

    Returns
    -------
    model
        Pytorch model.
    """
    if params.get('use_resnet', False):
        raise NotImplementedError

    layers = []
    current_dim = input_dim
    if params['n_dense_layers'] > 1:
        current_dim = params['dense_dim']
        for ii in range(params['n_dense_layers']-1):
            if ii == 0:
                layers.append(DropoutRescale(params['input_dropout'],
                                             params['input_rescale']))
                if settings.get('info', False):
                    layers.append(LinearInfoLognormal(input_dim, current_dim))
                else:
                    layers.append(Linear(input_dim, current_dim))
            else:
                layers.append(DropoutRescale(params['dense_dropout'],
                                             params['dense_rescale']))
                layers.append(Linear(current_dim, current_dim))
            layers.append(_activation[params['activation']]())
            if params['dense_dim_change'] != 'none':
                raise NotImplementedaccuracy

    layers.append(DropoutRescale(params['dense_dropout'],
                                 params['dense_rescale']))
    layers.append(Linear(current_dim, output_dim))

    model = SequentialAllActivations(*layers)
    return model

def make_conv2d_model(input_shape, output_shape, params, settings):
    """Create a densely connected model given hyperparameters.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    output_dim: int
        Number of output features/classes.
    params: dict
        Dictionary of model-specific hyperparameters.
    settings: dict
        Dictionary of model-common hyperparameters.

    Returns
    -------
    model
        Pytorch model.
    """
    resnet = params.get('use_resnet', False)
    batch_norm = params.get('use_batch_norm', False)

    layers = []

    out_filters = params['initial_kernel_number']
    out_filters = 10
    output = torch.zeros((2,) + input_shape)
    if params['n_conv_layers'] > 0:
        for ii in range(params['n_conv_layers']):
            if ii == 0:
                if params['conv_dropout_type'] == 'dropout':
                    layers.append(DropoutRescale(params['input_dropout'],
                                                 params['input_rescale']))
                    output = layers[-1].forward(output)
                elif params['conv_dropout_type'] == 'conv':
                    layers.append(Dropout2dRescale(params['input_dropout'],
                                                          params['input_rescale']))
                    output = layers[-1].forward(output)
                shp = output.shape[2:]
                kernel_size = params['input_kernel_size']
                kernel_size = (min(kernel_size[0], shp[0]),
                               min(kernel_size[1], shp[1]))
                layers.append(Conv2d(input_shape[0], out_filters,
                                     kernel_size))
                print(layers)
                output = layers[-1].forward(output)
                layers.append(_activation[params['activation']]())
                output = layers[-1].forward(output)
                if batch_norm:
                    layers.append(BatchNorm2d(out_filters))
                    output = layers[-1].forward(output)
            else:
                in_filters = out_filters
                if params['conv_dropout_type'] == 'dropout':
                    layers.append(DropoutRescale(params['conv_dropout'],
                                                 params['conv_rescale']))
                    output = layers[-1].forward(output)
                elif params['conv_dropout_type'] == 'conv':
                    layers.append(Dropout2dRescale(params['conv_dropout'],
                                                          params['conv_rescale']))
                    output = layers[-1].forward(output)
                else:
                    if params['conv_dropout_type'] != 'none':
                        raise ValueError
                if resnet:
                    layers.append(Conv2dResnetBlock(in_filters, out_filters))
                    output = layers[-1].forward(output)
                else:
                    if params['conv_dim_change'] == 'double':
                        out_filters = out_filters * 2
                    elif params['conv_dim_change'] == 'halve-first':
                        if ii == 0:
                            out_filters = out_filters // 2
                    elif params['conv_dim_change'] == 'halve-last':
                        if ii == params['n_conv_layers']-2:
                            out_filters = out_filters // 2
                    else:
                        if params['conv_dim_change'] != 'none':
                            raise ValueError


                    shp = output.shape[2:]
                    kernel_size = params['conv_kernel_size']
                    kernel_size = (min(kernel_size[0], shp[0]),
                                   min(kernel_size[1], shp[1]))
                    layers.append(Conv2d(in_filters, out_filters,
                                         kernel_size))
                    output = layers[-1].forward(output)
                    layers.append(_activation[params['activation']]())
                    output = layers[-1].forward(output)
                    if batch_norm:
                        layers.append(BatchNorm2d(out_filters))
                        output = layers[-1].forward(output)

                if params['pool_size'] is not None:
                    shp = output.shape[2:]
                    pool_size = params['pool_size']
                    pool_size = (min(kernel_size[0], shp[0]),
                                 min(kernel_size[1], shp[1]))
                    layers.append(MaxPool2d(pool_size))
                    output = layers[-1].forward(output)


    layers.append(Flatten())

    input_dim = np.prod(output.shape[1:])
    current_dim = params['dense_dim']
    if params['n_dense_layers'] > 1:
        for ii in range(params['n_dense_layers']-1):
            layers.append(DropoutRescale(params['dense_dropout'],
                                         params['dense_rescale']))
            if ii == 0:
                if settings.get('info', False):
                    layers.append(LinearInfoLognormal(input_dim, current_dim))
                else:
                    layers.append(Linear(input_dim, current_dim))
            else:
                layers.append(Linear(current_dim, current_dim))
            layers.append(_activation[params['activation']]())
            if batch_norm:
                layers.append(BatchNorm1d(current_dim))
            if params['dense_dim_change'] != 'none':
                raise NotImplementedError
    else:
        current_dim = input_dim

    layers.append(DropoutRescale(params['dense_dropout'],
                                 params['dense_rescale']))
    layers.append(Linear(current_dim, output_shape))

    model = Sequential(*layers)
    return model

def make_vae_model(input_shape, params):
    """Create a VAE model given hyperparameters.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    params: dict
        Dictionary of model-specific hyperparameters.

    Returns
    -------
    model
        Pytorch model.
    """
    layers = ModuleList()
    conv_layers = ModuleList()
    layer_settings = [] # conv: ['c', in, out, kernel]; fc: ['d', in, out]; max: ['m', size]
    activations = []   # params['activation']
    pool_indices = []
    out_filters = params['initial_kernel_number']
    output = torch.zeros((5,) + input_shape)
    if params['n_conv_layers'] > 0:
        for ii in range(params['n_conv_layers']):
            if ii == 0:
                shp = output.shape[2:]
                kernel_size = params['input_kernel_size']
                kernel_size = (min(kernel_size[0], shp[0]),
                               min(kernel_size[1], shp[1]))
                layer_settings.append(['c', input_shape[0], out_filters, kernel_size])
                layers.append(Conv2d(*layer_settings[-1][1:]))

                output = layers[-1].forward(output)
                nonLin = params['activation']
                layers.append(_activation[nonLin]())
                activations.append(nonLin)
                output = layers[-1].forward(output)
            else:
                in_filters = out_filters
                if params['conv_dim_change'] == 'double':
                    out_filters = out_filters * 2
                elif params['conv_dim_change'] == 'halve-first':
                    if ii == 0:
                        out_filters = out_filters // 2
                elif params['conv_dim_change'] == 'halve-last':
                    if ii == params['n_conv_layers']-2:
                        out_filters = out_filters // 2
                else:
                    if params['conv_dim_change'] != 'none':
                        raise ValueError
                shp = output.shape[2:]

                kernel_size = params['conv_kernel_size']
                kernel_size = (min(kernel_size[0], shp[0]),
                               min(kernel_size[1], shp[1]))
                layer_settings.append(['c', in_filters, out_filters, kernel_size])
                layers.append(Conv2d(*layer_settings[-1][1:]))
                output_temp = layers[-1].forward(output)
                nonLin = params['activation']
                layers.append(_activation[nonLin]())
                activations.append(nonLin)
                output_temp = layers[-1].forward(output_temp)
                shp_temp = output_temp.shape[2:]
                if shp_temp[0] < 3 or shp_temp[1] < 3:
                    layer_settings.pop()
                    activations.pop()
                    layers = layers[:-2]
                    break
                output = output_temp
                shp = shp_temp
                if params['pool_size'] is not None:
                    shp_temp = output.shape[2:]
                    pool_size = params['pool_size']
#                     hout = (shp[0] - kernel_size[0] + 1) % pool_size[0]
                    
                    pool_size = (min(pool_size[0], shp_temp[0]),
                                 min(pool_size[1], shp_temp[1]))
                    if shp_temp[0] in pool_size:
                        continue
                    else:
                        shp = shp_temp
                        layer_settings.append(['m', pool_size])
                        if len(layers) != 0:
                            conv_layers.append(layers)
                        layers = ModuleList()
                        layers.append(MaxPool2d(layer_settings[-1][1], return_indices=True))
                        output, indice = layers[-1].forward(output)
                        pool_indices.append(indice)
                        conv_layers.append(layers)
#                         layers = ModuleList()
#     if output.shape[2] > 7 or output.shape[3] > 7:
#         raise ValueError(f"Conv dimension not small enough: {output.shape[2:]}")                    
#     layers.append(Flatten())
    if len(layers) != 0:
        conv_layers.append(layers)
    enc_conv = ModuleList()
    if len(conv_layers) == 0:
        print('no maxpool')
        enc_conv = ModuleList(layers)
    else:
        for l in conv_layers:
            if type(l[0]) == MaxPool2d:
                enc_conv.append(*l)
            else:
                enc_conv.append(ModuleList(l))
    conv_layers = ModuleList()
    layers = ModuleList()
    conv_layer_settings = layer_settings
    layer_settings = []
    
    input_dim = np.prod(output.shape[1:])
    current_dim = params['dense_dim']
    
    in_first_last = [input_dim, output.size()]
    
    if params['n_dense_layers'] > 1:
        for ii in range(params['n_dense_layers']-1):
            if ii == 0:
                layer_settings.append(['d', input_dim, current_dim])
                layers.append(Linear(*layer_settings[-1][1:]))
            else:
                layer_settings.append(['d', current_dim, current_dim])
                layers.append(Linear(*layer_settings[-1][1:]))
            nonLin = params['activation']
            layers.append(_activation[nonLin]())
            activations.append(nonLin)
    else:
        current_dim = input_dim
    layer_settings.append(['d', current_dim, params['h_dim'] * 2])
    layers.append(Linear(*layer_settings[-1][1:]))
    
    enc_lin = ModuleList(layers)
    layers = ModuleList()
    lin_layer_settings = layer_settings
    
    lin_layer_settings.reverse()
    conv_layer_settings.reverse()
    
    layers = ModuleList()
    for i, layer in enumerate(lin_layer_settings):
        if i == 0:
            assert layer[0] == 'd'
            layers.append(Linear(params['h_dim'], layer[1]))
            nonLin = activations.pop()
            layers.append(_activation[nonLin]())
        elif layer[0] == 'd':
            layers.append(Linear(layer[2], layer[1]))
            nonLin = activations.pop()
            layers.append(_activation[nonLin]())
        else:
            raise ValueError
    
    dec_lin = ModuleList(layers)
    layers = ModuleList()
    
    for i, layer in enumerate(conv_layer_settings):
        if layer[0] == 'c':
            layers.append(ConvTranspose2d(layer[2], layer[1], layer[3]))
            if len(activations) != 0:
                nonLin = activations.pop()
                layers.append(_activation[nonLin]())            
        elif layer[0] == 'm':
            if len(layers) != 0: 
                conv_layers.append(layers)
            layers = ModuleList()
            indice = pool_indices.pop()
            layers.append(MaxUnpool2d(layer[1]))
            conv_layers.append(layers)
            layers = ModuleList()
    if len(layers) != 0:
        conv_layers.append(layers)
    dec_conv = ModuleList()

    for l in conv_layers:
        if type(l[0]) != MaxUnpool2d:
            dec_conv.append(ModuleList(l))
        else:
            dec_conv.append(*l)
    model = BVAE(params['h_dim'], enc_conv, enc_lin, dec_lin, dec_conv, in_first_last)
    return model