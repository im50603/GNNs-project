import torch
import torch_geometric
from atom3d.datasets import load_dataset
import argparse
import logging
import sys
import pickle
import csv

import data
from model import MultiLayerGraphRegressionModel

def eval(model, data_loader, criterion, device):
    outputs = []
    total_loss = 0
    num_batches = 0
    model.eval()
    for i, batch_data in enumerate(data_loader):
        y = batch_data.label.to(device) 
        out = model(batch_data, device)
        loss = criterion(out * 10, y.unsqueeze(1))
        total_loss += loss.item()
        num_batches += 1

        for j in range(len(batch_data.id)): 
            cur_out = out[j].item() * 10
            cur_target = y[j].item()
            cur_id = batch_data.id[j]
            cur_file_path = batch_data.file_path[j]
            outputs.append((cur_id, cur_file_path, cur_target, cur_out))
    
    avg_loss = total_loss / num_batches
    return avg_loss, outputs


def main():

    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument('--mpnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'mpnn'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device index (default: -1 for CPU)')
    
    hparams = parser.parse_args()  
    logger = logging.getLogger("lightning")

    config = {
        "mpnn_type": hparams.mpnn_type,
        "batch_size": hparams.batch_size,
        "in_node_channels": 3,
        "in_edge_channels": 3,
        "out_channels": 1, 
        "hidden_channels": [24, 12], #[24, 12,6],
        "num_layers": 1
    }

    if hparams.process:
        logger.info("Preprocessing the data...")
        if hparams.filetype == 'pdb':
            dataset = load_dataset(hparams.dataset_path, hparams.filetype, transform=data.add_scores)
        else: # expecting scores in lmdb files
            dataset = load_dataset(hparams.dataset_path, hparams.filetype)
        data_list = data.transform_dataset(dataset)
    else:
        logger.info(f"Loading preprocessed datasets...")
        with open(hparams.dataset_path, 'rb') as f:
            data_list = pickle.load(f)
    
    dataloader = torch_geometric.loader.DataLoader(
        data_list,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)

    # TRAINING
    aggr = torch_geometric.nn.aggr.MeanAggregation()
    model = MultiLayerGraphRegressionModel(config['in_node_channels'], config['in_edge_channels'], config['hidden_channels'], config['out_channels'], config["mpnn_type"], aggr, config["num_layers"])
    model.load_state_dict(torch.load(hparams.model_file))
    
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda:{}'.format(hparams.gpu) if torch.cuda.is_available() and hparams.gpu >= 0 else 'cpu')
    model.to(device)
    logger.info(f"Chosen device: {device}")
    logger.info("Running evaluation...")
    
    loss, outputs = eval(model, dataloader, criterion, device)
    print('Loss:', loss)
    with open(hparams.output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "file_path", "rms", "out"])
        writer.writerows(outputs)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    
    main()


