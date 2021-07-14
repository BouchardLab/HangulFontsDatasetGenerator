import os, pickle
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from pathlib import Path
from math import ceil

from hangul_analysis.nets.model_creator import (make_vae_model,
                                                create_summary_writer)
from hangul_analysis.vaes.functions import kl_div

def vae_train_loop(ds, params, recon, base, base_valid, base_data,
                   model_id, fold, beta, device='cpu', pretrained='',
                   ex=0, seed=0):
    """
    Train, validate, and save a model.

        Parameters
        ----------
        ds
            Train, valid, and test data loaders.
        others
            Same as in main
    """
    ds_train, ds_valid, ds_test = ds
    temp_x = ds_train.dataset.tensors[0][0].clone().detach()
    in_size = temp_x.unsqueeze(0).size()
    in_size = (in_size[0], in_size[1], in_size[2])
    args = (in_size, params)
    model = make_vae_model(*args).to(device)
    if pretrained != '':
        root = pretrained
        ps = list(Path(root).rglob('c*.pt'))
        model.load_state_dict(torch.load(ps[0], map_location=device))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if beta and total_params < 200000:
        raise ValueError(f"Network too small, {total_params} total parameters")
    elif beta and total_params > 2000000:
        raise ValueError(f"Network too large, {total_params} total parameters")
    elif not beta and total_params > 4000000:
        raise ValueError(f"Network too large, {total_params} total")
    print(f'Total params: {total_params}')
    epochs_mini = 5
    epochs = 25 * epochs_mini
    C = params['C']
    count = 0
    gamma = params['gamma']
    h_dim = params['h_dim']
    batch_size = params['batch_size']
    gap = torch.ones(8, *in_size).to(device)
    print(device)
    with create_summary_writer(model, ds_train, base_data, model_id, device=device, conv=True) as writer:
        with open(os.path.join(base_data, model_id, 'model_params.pkl'), 'wb') as f:
            pickle.dump(args, f)
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=mom, weight_decay=wd)
        sched = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        def train_step(engine, batch):
            model.train()
            data, _ = batch
            data = data.unsqueeze(1)
            data = data.to(device)
            optimizer.zero_grad()
            count = engine.state.count
            ind = count
            if count < 0:
                ind = 0
            c = C[ind]
            gam = gamma[ind]
            reconstruction, mu, logvar = model.forward(data)
            bce_loss = model.bce_loss(reconstruction, data)
            if not beta:
                kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                             require_grad=True, device=device)
            else:
                kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                             require_grad=True, device=device)
            if count < 0:
                    kld = torch.tensor([0.]).to(device)
            loss = kld + bce_loss
            running_loss = loss.item()
            loss.backward()
            optimizer.step()
            return bce_loss.item(), kld.item(), running_loss
        trainer = Engine(train_step)

        def train_eval_step(engine, batch):
            model.eval()
            running_loss = 0.0
            bce_loss = 0.0
            kld = 0.0
            with torch.no_grad():
                data, _ = batch
                data = data.unsqueeze(1)
                data = data.to(device)
                count = engine.state.count
                ind = count
                if count < 0:
                    ind = 0
                c = C[ind]
                gam = gamma[ind]
                reconstruction, mu, logvar = model.forward(data)
                bce_loss = model.bce_loss(reconstruction, data)
                if not beta:
                    kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                                 device=device)
                else:
                    kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                                 device=device)
                if count < 0:
                    kld = torch.tensor([0.]).to(device)
                loss = kld + bce_loss
                running_loss = loss.item()
            return bce_loss.item(), kld.item(), running_loss, data, reconstruction
        train_evaluator = Engine(train_eval_step)

        def validation_step(engine, batch):
            model.eval()
            running_loss = 0.0
            reconstruction = 0.0
            data = 0.0
            bce_loss = 0.0
            kld = 0.0
            with torch.no_grad():
                data, _ = batch
                data = data.unsqueeze(1)
                data = data.to(device)
                count = engine.state.count
                ind = count
                if count < 0:
                    ind = 0
                c = C[ind]
                gam = gamma[ind]
                reconstruction, mu, logvar = model.forward(data)
                bce_loss = model.bce_loss(reconstruction, data)
                if not beta:
                    kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                                device=device)
                else:
                    kld = kl_div(h_dim, torch.cat((mu, logvar), dim=1), hp=[gam, c],
                                 device=device)
                if count < 0:
                    kld = torch.tensor([0.]).to(device)
                loss = kld + bce_loss
                running_loss = loss.item()
            return bce_loss.item(), kld.item(), running_loss, data, reconstruction
        valid_evaluator = Engine(validation_step)

        @trainer.on(Events.STARTED)
        def start_message():
            print("Start training!")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation(engine):
            valid_evaluator.run(recon)
            bce, kl, total, data, reconstruction = valid_evaluator.state.output
            bs = data.size(0)
            num_rows = min(bs, 64)
            pad = torch.zeros(((ceil(num_rows/8) * 8) - num_rows, *in_size)).to(device)
            both = torch.cat((data.view(bs, *in_size)[:num_rows], pad, gap,
                              reconstruction.view(bs, *in_size)[:num_rows]))
            writer.add_images("validation_chosen/C_{}_{}.png"
                              .format((engine.state.epoch - 1) // epochs_mini,
                                      (engine.state.epoch - 1) % epochs_mini),
                                      both.cpu(), engine.state.epoch)
            writer.add_scalar("validation_chosen/total_loss",
                              total, engine.state.epoch)
            writer.add_scalar("validation_chosen/bce_loss", bce, engine.state.epoch)
            writer.add_scalar("validation_chosen/kl_loss", kl, engine.state.epoch)
            valid_evaluator.run(ds_valid)
            bce, kl, total, data, reconstruction = valid_evaluator.state.output
            print("Validation - Epoch: {} Total Loss: {} BCE: {} KL: {}"
                  .format(engine.state.epoch, total, bce, kl))
            if engine.state.epoch == 1:
                print("valid", data.shape, reconstruction.shape)
            bs = data.size(0)
            num_rows = min(bs, 64)
            pad = torch.zeros(((ceil(num_rows/8) * 8) - num_rows, *in_size)).to(device)
            both = torch.cat((data.view(bs, *in_size)[:num_rows], pad, gap,
                              reconstruction.view(bs, *in_size)[:num_rows]))
            writer.add_images("validation/C_{}_{}.png"
                              .format((engine.state.epoch - 1) // epochs_mini,
                                      (engine.state.epoch - 1) % epochs_mini),
                                      both.cpu(), engine.state.epoch)
            writer.add_scalar("validation/total_loss",
                              total, engine.state.epoch)
            writer.add_scalar("validation/bce_loss", bce, engine.state.epoch)
            writer.add_scalar("validation/kl_loss", kl, engine.state.epoch)
            if beta:
                if engine.state.epoch == 5 and fold == 0 and bce > 0.04:
                    raise RuntimeError("reconstruction loss too high")

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            bce, kl, total, _, _ = valid_evaluator.state.output
            sched.step(total)

        @trainer.on(Events.ITERATION_COMPLETED(every=500))
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch),
                            batch_size=batch_size)
            train_evaluator.run(ds)
            bce, kl, total, _, _ = train_evaluator.state.output
            writer.add_scalar("batchtrain/total_loss",
                              total, engine.state.iteration)
            writer.add_scalar("batchtrain/bce_loss", bce,
                              engine.state.iteration)
            writer.add_scalar("batchtrain/kl_loss", kl,
                              engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_lr(engine):
            writer.add_scalar(
                "lr", optimizer.param_groups[0]['lr'], engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(ds_train)
            bce, kl, total, data, reconstruction = train_evaluator.state.output
            print("Training Results - Epoch: {} Total Loss: {} BCE: {} KL: {}"
                  .format(engine.state.epoch, total, bce, kl))
            bs = data.size(0)
            num_rows = min(bs, 64)
            pad = torch.zeros(((ceil(num_rows/8) * 8) - num_rows, *in_size)).to(device)
            both = torch.cat((data.view(bs, *in_size)[:num_rows], pad, gap,
                              reconstruction.view(bs, *in_size)[:num_rows]))
            writer.add_images("train/C_{}_{}.png"
                            .format((engine.state.epoch - 1) // epochs_mini,
                                      (engine.state.epoch - 1) % epochs_mini),
                                      both.cpu(), engine.state.epoch)
            writer.add_scalar("train/total_loss",
                              total, engine.state.epoch)
            writer.add_scalar("train/bce_loss", bce, engine.state.epoch)
            writer.add_scalar("train/kl_loss", kl, engine.state.epoch)

        @trainer.on(Events.STARTED)
        def create_c(engine):
            engine.state.count = -2
            train_evaluator.state.count = -2
            valid_evaluator.state.count = -2
            if pretrained != '':
                engine.state.count = 0
                train_evaluator.state.count = 0
                valid_evaluator.state.count = 0

        @trainer.on(Events.EPOCH_COMPLETED(every=epochs_mini))
        def update_c(engine):
            engine.state.count += 1
            train_evaluator.state.count += 1
            valid_evaluator.state.count += 1

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_value(engine):
            temp = engine.state.output
            return -temp[2]

        @trainer.on(Events.EPOCH_COMPLETED)
        def epoch_value(engine):
            return engine.state.epoch

        checkpoint = ModelCheckpoint(os.path.join(base_data, model_id), model_id,
                                     score_function=validation_value)
        checkpoint_latest = ModelCheckpoint(os.path.join(base_data, model_id), f"c_{model_id}")

        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint, {'model': model})
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_latest, {'model': model})

        # kick everything off
        trainer.run(ds_train, max_epochs=epochs)
