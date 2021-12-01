from model import GraphFormer
from data import GraphDataModule, get_dataset
import os
import torch
import poptorch
import math
import time
from tqdm import tqdm
import argparse
from lr import PolynomialDecayLR

####TESTING
from collator import collator
from functools import partial
import ogb
import ogb.lsc
import ogb.graphproppred
from ogb_wrapper import MyPygPCQM4MDataset2

import wandb
from ipu_utils import pipeline_model

from datetime import datetime

wandb.init(project="graphormer-project", entity="graphormer")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    print(str(datetime.now()))

### ARGUMENTS
    parser = argparse.ArgumentParser(description='Get batch size related parameters.')
    parser.add_argument('--mini_batch_size', type=int)
    parser.add_argument('--gradient_accumulation', type=int)
    parser.add_argument('--replication_factor', type=int)
    parser.add_argument('--device_iteration', type=int)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--pipeline_splits', type=str, nargs='+', default=[], help="List of the splitting layers")
    parser.add_argument('--test', type=int, default=0, help ='0 for train, 1 for test')

    parser.add_argument('--checkpoint_file', type=str, default="", help="which checkpoint file to load?")
    args = parser.parse_args()
### ARGUMENTS


#### MODEL CONFIGURATION
    n_layers=12
    head_size=32
    hidden_dim=768
    dropout_rate=0.0
    intput_dropout_rate=0.0
    weight_decay=0.0
    ffn_dim=768
    dataset_name='PCQM4M-LSC'
    warmup_updates = 10000
    tot_updates = 1500000
    peak_lr = 2e-4
    end_lr = 1e-09
    edge_type='multi_hop'
    multi_hop_max_dist=20
    attention_dropout_rate=0.1
    flag=False
    flag_m=3
    flag_step_size=1e-3
    flag_mag=1e-3

    GRAD_ACCUL = args.gradient_accumulation
    REP_FAC = args.replication_factor
    MINI_BATCH_SIZE = args.mini_batch_size
    DEV_ITER = args.device_iteration
    GLOBAL_BATCH_SIZE = GRAD_ACCUL * REP_FAC * MINI_BATCH_SIZE
### MODEL CONFIGURATION


### CONSTRUCT MODEL
    model = GraphFormer(n_layers, head_size, hidden_dim, dropout_rate, intput_dropout_rate, weight_decay, ffn_dim, dataset_name, warmup_updates, tot_updates, peak_lr, 
    end_lr, edge_type, multi_hop_max_dist, attention_dropout_rate, flag, flag_m, flag_step_size, flag_mag)
    print('total params:', sum(p.numel() for p in model.parameters()))
    pipeline_model(model, args.pipeline_splits)
### CONSTRUCT MODEL


### SET UP OPTIMIZER
    # increase learning rate by sqrt(new_batch/old_batch)
    peak_lr = 2e-4
    end_lr = 1e-09

    new_peak_lr = math.sqrt(GLOBAL_BATCH_SIZE/256) * peak_lr
    new_end_lr = math.sqrt(GLOBAL_BATCH_SIZE/256) * end_lr

    print(f'peak_lr = {new_peak_lr}')
    print(f'end_lr = {new_end_lr}')

    totalWarmupSamples = 256 * 10000
    totalProcessedSamples = 256 * 1500000

    new_warmup_updates = int(totalWarmupSamples / GLOBAL_BATCH_SIZE)
    new_total_updates = int(totalProcessedSamples / GLOBAL_BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=new_peak_lr, weight_decay=0.0)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=new_warmup_updates, tot_updates=new_total_updates, lr=new_peak_lr, end_lr=new_end_lr, power=1.0)
### SET UP OPTIMIZER


### CHECKPOINT
    if args.checkpoint:
        CHECKPOINT_PATH = './checkpoint/' + args.checkpoint_file
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
### CHECKPOINT


### POPTORCH OPTIONS
    training_opts = poptorch.Options()

    print(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')
    print(f'DEV_ITER = {DEV_ITER}')

    training_opts.deviceIterations(DEV_ITER)
    training_opts.Training.gradientAccumulation(GRAD_ACCUL)
    training_opts.replicationFactor(REP_FAC)
    inference_opts = poptorch.Options().replicationFactor(args.replication_factor).deviceIterations(args.gradient_accumulation)
### POPTORCH OPTIONS


### WRAP MODEL TO POPTORCH
    model.autocast()
    training_model = poptorch.trainingModel(model, training_opts, optimizer)
    # inference_model = poptorch.inferenceModel(model, inference_opts)
### WRAP MODEL TO POPTORCH


### DATA LOADER
    dataModule = GraphDataModule(dataset_name='PCQM4M-LSC', num_workers=16, batch_size=MINI_BATCH_SIZE, seed=0, multi_hop_max_dist=20, rel_pos_max=1024, training_opts=training_opts)
    dataModule.setup()

    def train_dataloader():
        loader = poptorch.DataLoader(
            training_opts,
            dataModule.dataset_train,
            batch_size=MINI_BATCH_SIZE,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader():
        loader = poptorch.DataLoader(
            inference_opts,
            dataModule.dataset_val,
            batch_size=MINI_BATCH_SIZE,
            shuffle=False,
            num_workers=32,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader():
        loader = poptorch.DataLoader(
            inference_opts,
            dataModule.dataset_test,
            batch_size=MINI_BATCH_SIZE,
            shuffle=False,
            num_workers=32,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(test_dataloader)', len(loader))
        return loader

    train_loader = train_dataloader()
    val_loader = val_dataloader()
    test_loader = test_dataloader()
    print(len(train_loader))
### DATA LOADER


## TRAIN START 
    if not args.test:
        N_EPOCHS = int(new_total_updates / len(train_loader))
        print(N_EPOCHS)
        model.train()

    #EPOCH START
        for epoch in range(N_EPOCHS):
            wandb.log({"epoch": epoch})

            loss_accumulate = 0
            throughput_accumulate = 0
            counter = 0

            epoch_loss_accumulate = 0

        ### TRAINING LOOP START
            with tqdm(train_loader, unit="batch") as tepoch:
                for batched_data in tepoch:
                    
                    start_time = time.time()
                    output, loss = training_model(batched_data)
                    end_time = time.time()

                    lr_scheduler.step()
                    training_model.setOptimizer(optimizer)

                    tepoch.set_postfix(loss = torch.mean(loss).item())

                    loss_accumulate += torch.mean(loss).item()
                    throughput_accumulate += ( GLOBAL_BATCH_SIZE / (end_time - start_time) )
                    counter += 1

                    if counter % 100 == 0:
                        wandb.log({"train_loss": loss_accumulate/100, "learning_rate":lr_scheduler.get_last_lr()[0], "throughput (samples/s)" : (throughput_accumulate/100)})
                        print(throughput_accumulate/100)
                        epoch_loss_accumulate += loss_accumulate
                        throughput_accumulate = 0
                        counter = 0
                        loss_accumulate = 0

            epoch_loss = epoch_loss_accumulate / len(train_loader)
            wandb.log({"epoch_loss" : epoch_loss})
        ### TRAINING LOOP END

        #     training_model.detachFromDevice()
        #     model.eval()
        #     total_valid_loss = 0

        # ### VALIDATION LOOP START
        #     with tqdm(val_loader, unit='batch') as vepoch:
        #         for batched_data in vepoch:
        #             output, valid_loss = inference_model(batched_data)
        #             total_valid_loss += torch.mean(valid_loss).item()
        # ### VALIDATION LOOP END

        #     inference_model.detachFromDevice()
        #     model.train()

            if (epoch) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'lr_scheduler_state_dict':lr_scheduler.state_dict(),
                }, f'./checkpoint/checkpoint{str(epoch) + str(args.mini_batch_size) + str(args.gradient_accumulation) + str(args.replication_factor) + str(args.device_iteration)}.pt',)

        #     val_loss = total_valid_loss / len(val_loader)
        #     wandb.log({"valid_mae": val_loss})
    ### EPOCH END
        torch.save(model, './savedModel/GraphormerIPU.pth')
        print('Training Complete. Model Saved.')
### TRAIN END


### TEST START
    if args.test:
        TEST_MODEL_PATH = './savedModel/GraphormerIPU.pth' 
        model = torch.load(TEST_MODEL_PATH)
        model.eval()
        total_test_loss = 0

        throughput_accumulate = 0

        inference_model = poptorch.inferenceModel(model, inference_opts)

        with tqdm(val_loader, unit='batch') as testRound:
            for batched_data in testRound:
                start_time = time.time()
                output, loss = inference_model(batched_data)
                end_time = time.time()

                total_test_loss += torch.mean(loss).item()
                throughput_accumulate += (GLOBAL_BATCH_SIZE / (end_time - start_time))

        wandb.log({"test_loss": total_test_loss/len(val_loader)})
        print(f'throughput (samples/s) = {throughput_accumulate/ len(val_loader)}')
        print(f'test loss MAE= {total_test_loss / len(val_loader)}')
### TEST END