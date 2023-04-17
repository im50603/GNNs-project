import torch
import torch_geometric
import argparse
import pickle
from atom3d.datasets import load_dataset
import wandb
import logging
import sys

from model import GraphRegressionModel, MultiLayerGraphRegressionModel
import data

wandb_log_train_step = 0
wandb_log_val_step = 0

def train_step(model, train_loader, optimizer, criterion, device, track_gradients=False):
    global wandb_log_train_step
    total_loss = 0
    num_batches = 0
    model.train()
    for i, batch_data in enumerate(train_loader):
        y = batch_data.label.to(torch.float32).to(device) 

        optimizer.zero_grad()
        out = model(batch_data, device)
        loss = criterion(out * 10, y.unsqueeze(1))
        total_loss += loss.item()
        num_batches += 1

        wandb.log({"train_step": wandb_log_train_step, "Train Loss": loss.item()})
        wandb_log_train_step += 1

        loss.backward()  
        optimizer.step()

        if track_gradients:
            # Log gradients to Weights & Biases
            for name, param in model.named_parameters():
                if param.grad is not None: # edge_linear layer has no grad when the gnn model is gcn as it does not use egde features
                    wandb.log({"Gradients/" + name: param.grad.norm().item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def eval_step(model, data_loader, criterion, device):
    global wandb_log_val_step

    total_loss = 0
    num_batches = 0
    model.eval()
    for i, batch_data in enumerate(data_loader):
        y = batch_data.label.to(device) 

        out = model(batch_data, device)
        loss = criterion(out * 10, y.unsqueeze(1))

        wandb.log({"val_step": wandb_log_val_step, "Val Loss": loss.item()})
        wandb_log_val_step += 1

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100):
    best_model_state_dict = None
    best_val_loss = float('inf')

    wandb.watch(model)

    for epoch in range(num_epochs):
        # Training
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        wandb.log({"epoch": epoch, "Avg Train Loss": train_loss}) #, step=epoch)

        # Validation
        val_loss = eval_step(model, val_loader, criterion, device)
        wandb.log({"epoch": epoch, "Avg Val Loss": val_loss}) #, step=epoch)

        # Check if the current model has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()

#        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Load the best model state dict
    model.load_state_dict(best_model_state_dict)

    # Return the best model and the train and val losses
    return model


def main():
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument('--mpnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'mpnn'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device index (default: -1 for CPU)')
    parser.add_argument('--model_file', type=str, default=None)

    hparams = parser.parse_args()  

    config = {
        "mpnn_type": hparams.mpnn_type,
        "learning_rate": hparams.learning_rate,
        "batch_size": hparams.batch_size,
        "num_epochs": hparams.num_epochs,
        "in_node_channels": 3,
        "in_edge_channels": 3,
        "hidden_channels": [24, 12], # [48, 24, 12, 6]
        "out_channels": 1, 
        "num_layers": 1
    }
    wandb.init(project="gnn_project", config=config) #, name="my_experiment")
    logger = logging.getLogger("lightning")


    # DATA PREP
    if hparams.process:
        logger.info("Preprocessing the data...")
        if hparams.filetype == 'pdb':
            train_dataset = load_dataset(hparams.train_dataset, hparams.filetype, transform=data.add_scores)
            val_dataset = load_dataset(hparams.val_dataset, hparams.filetype, transform=data.add_scores)
        else: # expecting scores in lmdb files
            train_dataset = load_dataset(hparams.train_dataset, hparams.filetype)
            val_dataset = load_dataset(hparams.val_dataset, hparams.filetype)
        train_data_list = data.transform_dataset(train_dataset)
        val_data_list = data.transform_dataset(val_dataset) 
    else:
        logger.info(f"Loading preprocessed datasets...")
        with open(hparams.train_dataset, 'rb') as f:
            train_data_list = pickle.load(f)
        with open(hparams.val_dataset, 'rb') as f:
            val_data_list = pickle.load(f)
    
    train_dataloader = torch_geometric.loader.DataLoader(
        train_data_list,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)

    val_dataloader = torch_geometric.loader.DataLoader(
        val_data_list,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    # TRAINING
    aggr = torch_geometric.nn.aggr.MeanAggregation()
#    model = GraphRegressionModel(config['in_node_channels'], config['in_edge_channels'], config['hidden_channels'], config['out_channels'], config["mpnn_type"], aggr)
    model = MultiLayerGraphRegressionModel(config['in_node_channels'], config['in_edge_channels'], config['hidden_channels'], config['out_channels'], config["mpnn_type"], aggr, config["num_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda:{}'.format(hparams.gpu) if torch.cuda.is_available() and hparams.gpu >= 0 else 'cpu')
    model.to(device)
    logger.info(f"Chosen device: {device}")
    wandb.config.device = device

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_params}")
    wandb.config.num_params = num_params

    torch.autograd.set_detect_anomaly(True)
    logger.info("Running training...")
    model = train(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=config["num_epochs"])

    # SAVE RESULTS
    if hparams.model_file is not None:
        print("Saving the model...")
        torch.save(model.state_dict(), hparams.model_file)



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    
    main()


