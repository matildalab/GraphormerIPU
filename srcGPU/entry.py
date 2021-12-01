from model import GraphFormer
from data import GraphDataModule, get_dataset
import os
import torch

import math
import time
from tqdm import tqdm
import argparse
from lr import PolynomialDecayLR
from torch.utils.data import DataLoader


####TESTING
from collator import collator
from functools import partial
import ogb
import ogb.lsc
import ogb.graphproppred
from ogb_wrapper import MyPygPCQM4MDataset2

import wandb


from datetime import datetime

# wandb.init(project="graphormer-project-ipu", entity="graphormer", id="212nv9rx", resume="must")
wandb.init(project="graphormer-project", entity="graphormer")



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    print(str(datetime.now()))

    parser = argparse.ArgumentParser(description='Get batch size related parameters.')
    parser.add_argument('--mini_batch_size', type=int)
    parser.add_argument('--gradient_accumulation', type=int)
    parser.add_argument('--replication_factor', type=int)
    parser.add_argument('--device_iteration', type=int)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--pipeline_splits', type=str, nargs='+', default=[], help="List of the splitting layers")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### MODEL CONFIGURATION
    n_layers=12
    head_size=32
    hidden_dim=768
    dropout_rate=0.0
    intput_dropout_rate=0.0
    # weight_decay=0.0
    ffn_dim=768
    dataset_name='PCQM4M-LSC'
    # warmup_updates=10000
    # tot_updates=1500000
    edge_type='multi_hop'
    multi_hop_max_dist=20
    attention_dropout_rate=0.1
    flag=False
    # flag_m=3
    # flag_step_size=1e-3

    GRAD_ACCUL = args.gradient_accumulation
    REP_FAC = args.replication_factor
    MINI_BATCH_SIZE = args.mini_batch_size
    DEV_ITER = args.device_iteration
    GLOBAL_BATCH_SIZE = GRAD_ACCUL * REP_FAC * MINI_BATCH_SIZE
### MODEL CONFIGURATION

### CONSTRUCT MODEL
    model = GraphFormer(n_layers, head_size, hidden_dim, dropout_rate, intput_dropout_rate, ffn_dim, dataset_name, edge_type, multi_hop_max_dist, attention_dropout_rate, flag).to(device)
    print('total params:', sum(p.numel() for p in model.parameters()))
    
### CONSTRUCT MODEL


### SET UP OPTIMIZER
    # increase learning rate by sqrt(new_batch/old_batch)
    peak_lr = 2e-4
    end_lr = 1e-09

    # new_peak_lr = math.sqrt(GLOBAL_BATCH_SIZE/256) * peak_lr
    # new_end_lr = math.sqrt(GLOBAL_BATCH_SIZE/256) * end_lr

    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, weight_decay=0.0)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=10000, tot_updates=1500000, lr=peak_lr, end_lr=end_lr, power=1.0)
### SET UP OPTIMIZER

### CHECKPOINT
    CHECKPOINT_PATH = './checkpoint'
    print(args.checkpoint)
    if args.checkpoint:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
### CHECKPOINT

### POPTORCH OPTIONS
### POPTORCH OPTIONS


### DATA LOADER
    dataModule = GraphDataModule(dataset_name='PCQM4M-LSC', num_workers=16, batch_size=MINI_BATCH_SIZE, seed=0, multi_hop_max_dist=20, rel_pos_max=1024)
    dataModule.setup()

    def train_dataloader():
        loader = DataLoader(
            dataModule.dataset_train,
            batch_size=MINI_BATCH_SIZE,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader():
        loader = DataLoader(
            dataModule.dataset_val,
            batch_size=MINI_BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader():
        loader = DataLoader(
            dataModule.dataset_test,
            batch_size=MINI_BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset('PCQM4M-LSC')['max_node'], multi_hop_max_dist=20, rel_pos_max=1024),
        )
        print('len(test_dataloader)', len(loader))
        return loader
### DATA LOADER

    train_loader = train_dataloader()
    val_loader = val_dataloader()
    print(len(train_loader))

    N_EPOCHS = int(1500000 / len(train_loader))

    model.train()
    for epoch in range(N_EPOCHS):
        wandb.log({"epoch": epoch})

        epoch_start_time = time.time()
        epoch_loss = 0

        # throughput_accumulate = 0
        # counter = 0

### TRAINING LOOP START
        with tqdm(train_loader, unit="batch") as tepoch:
            for batched_data in tepoch:
                batched_data.to(device)
                optimizer.zero_grad()
                # start_time = time.time()
                output, loss = model(batched_data)
                loss.backward()
                optimizer.step()
                # end_time = time.time()
                epoch_loss += loss.item()
                wandb.log({"train_loss": loss.item()})
                
                # throughput_accumulate += ( GLOBAL_BATCH_SIZE / (end_time - start_time) )
                # counter += 1
                # if counter % 100 == 0:
                #     print(f'Throughput (samples/s) = {throughput_accumulate/100}')
                #     throughput_accumulate = 0
                #     counter = 0
                lr_scheduler.step()

        epoch_avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})
        #wandb.log({"Epoch Average Train Loss": epoch_avg_train_loss})
### TRAINING LOOP END
        
        model.eval()
        total_valid_loss = 0

### VALIDATION LOOP START
        with tqdm(val_loader, unit='batch') as vepoch:
            for batched_data in vepoch:
                batched_data.to(device)
                output, valid_loss = model(batched_data)
                total_valid_loss += torch.mean(valid_loss).item()
### VALIDATION LOOP END

        epoch_end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start_time, epoch_end_time)

        model.train()
        if (epoch) % 10 == 0 :
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_avg_train_loss,
                }, f'./checkpoint/checkpoint{epoch}.pt',)


        val_loss = total_valid_loss / len(val_loader)
        wandb.log({"valid_mae": val_loss})

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_avg_train_loss:.3f} | Train PPL: {math.exp(epoch_avg_train_loss):.3f}')
        print(f'\tValid Loss: {val_loss:.3f}')
