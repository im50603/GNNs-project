import torch
import torch_geometric
import atom3d
import pandas as pd
import numpy as np
from atom3d.datasets import load_dataset
import pickle

def prepare(item, k=50, label_to_use='rms'):
  element_mapping = {
    'C': 0, 
    'O': 1, 
    'N': 2,
   # 'P': 3,
  }
  num_channels = len(element_mapping)
  if type(item['atoms']) != pd.DataFrame:
        item['atoms'] = pd.DataFrame(**item['atoms'])
  coords = item['atoms'][['x', 'y', 'z']].values
  elements = item['atoms']['element'].values

  if label_to_use is None:
      # Don't use any label.
      label = [0]
  else:
      scores = item['scores']
      if type(scores) != pd.Series and 'data' in scores \
              and 'index' in scores:
          scores = pd.Series(
              scores['data'], index=scores['index'], name=item['id'])
      else:
          scores = pd.Series(scores, index=scores.keys(), name=item['id'])
      label = [scores[label_to_use]]

  sel = np.array([i for i, e in enumerate(elements) if e in element_mapping])
  total_atoms = elements.shape[0]
  coords = coords[sel]
  elements = elements[sel]

  # Make one-hot
  elements_int = np.array([element_mapping[e] for e in elements])
  one_hot = np.zeros((elements.size, len(element_mapping)))
  one_hot[np.arange(elements.size), elements_int] = 1

  geometry = torch.tensor(coords, dtype=torch.float32)
  features = torch.tensor(one_hot, dtype=torch.float32)
  label = torch.tensor(label)

  ra = geometry.unsqueeze(0)
  rb = geometry.unsqueeze(1)
  pdist = (ra - rb).norm(dim=2)
  tmp = torch.topk(-pdist, k, axis=1)

  nei_list = []
  geo_list = []
  for source, x in enumerate(tmp.indices):
    cart = geometry[x]
    nei_list.append(
    torch.tensor(
                [[source, dest] for dest in x], dtype=torch.long))
    geo_list.append(cart - geometry[source])
  nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
  geo_list = torch.cat(geo_list, dim=0)

  r_max = 10  # Doesn't matter since we override - not used here, only in original paper
  d = torch_geometric.data.Data(x=features, edge_index=nei_list, edge_attr=geo_list, pos=geometry) # for now ignore, useful for layers with spherical harmonics: , Rs_in=[(num_channels, 0)])
  d.r_max = r_max
  d.label = label
  d.id = item['id']
  d.file_path = item['file_path']

  return d


def transform_dataset(dataset, label_to_use=None):
  data_list = [prepare(item, label_to_use=label_to_use) for item in dataset]
  return data_list


def add_scores(item):
  pdb_path = item['file_path']
  scores = {}

  with open(pdb_path, 'r') as file:
    ter_found = False
    for line in file:
      if line.startswith("TER"):
        ter_found = True
      elif ter_found:
        line = line.strip()
        if line:
          tokens = line.split()
          key = tokens[0]
          value = float(tokens[1])
          scores[key] = value

  item['scores'] = scores
  return item


def main():
  train_dataset_path = '/home/martinovici/scratch/GNN_project/data/new_train/' # classics_train_val/lmdbs/example_train/'
  val_dataset_path = '/home/martinovici/scratch/GNN_project/data/new_val/' # classics_train_val/lmdbs/example_val/'
  out_train_transformed = 'data/processed/new_train_graphs.pkl'
  out_val_trainsformed = 'data/processed/new_val_graphs.pkl'
  filetype = 'pdb' # 'lmdb'
  
  # atom3d package loads_dataset from the path, and creates dataset based on the filetype -> PDBDataset, LMDBDataset, etc
  # IMPORTANT! If filetype == 'pdb' load with train_dataset = load_dataset(train_dataset_path, filetype_train, transform=add_scores)
  train_dataset = load_dataset(train_dataset_path, filetype, transform=add_scores)
  # after loading the dataset, we have to transform it, it can be done by adding transform function as a parameter of load_dataset function, 
  # but that way it is applied every time you access the data, so it becomes too slow
  train_data_list = transform_dataset(train_dataset, label_to_use='rms')
  with open(out_train_transformed, 'wb') as f:
    pickle.dump(train_data_list, f)

  val_dataset = atom3d.datasets.load_dataset(val_dataset_path, filetype, transform=add_scores)
  val_data_list = transform_dataset(val_dataset, label_to_use='rms') 

  with open(out_val_trainsformed, 'wb') as f:
    pickle.dump(val_data_list, f)


if __name__=='__main__':
  main2()

