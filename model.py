import torch
import torch_geometric as tg


class MPNN(tg.nn.MessagePassing):
  def __init__(self, in_channels, out_channels, edge_dim, aggr='mean'): #in_channels is dim of node features
    super(MPNN, self).__init__(aggr=aggr) 
    self.edge_dim = edge_dim
    M_in_channels = in_channels * 2 + edge_dim
    self.M = torch.nn.Linear(M_in_channels, out_channels)
    U_in_channels = in_channels + out_channels
    self.U = torch.nn.Linear(U_in_channels, out_channels)
    self.relu = torch.nn.ReLU()

  def forward(self, x, edge_index, edge_attr):
    return self.propagate(edge_index, x=x, edge_attr=edge_attr)

  def message(self, x_i, x_j, edge_attr): # x_i and x_j are node feature matrices of shape (num_edges, input_node_features_dim)
    edge_attr = edge_attr.view(-1, self.edge_dim) # when having only one edge feature, shape will be (num_edges), so it has to be transformed
    input = torch.cat([x_i, x_j, edge_attr], dim=-1)
    return self.relu(self.M(input))  # returns matrix of shape (num_edges, out_channels) which is aggregated with aggr function in aggr_out

  def update(self, aggr_out, x): # aggr_out has shape (num_nodes, out_channels)
    input = torch.cat([x, aggr_out], dim=-1)
    return self.relu(self.U(input))


class NodeFeatureLinear(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(NodeFeatureLinear, self).__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels)

  def forward(self, x):
    return torch.nn.functional.relu(self.linear(x))


class EdgeFeatureLinear(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(EdgeFeatureLinear, self).__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels)

  def forward(self, x):
    return torch.nn.functional.relu(self.linear(x))


class RegressionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegressionHead, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)


class GraphRegressionModel(torch.nn.Module):
  def __init__(self, in_node_channels, in_edge_channels, hidden_channels, out_channels, mpnn_type, aggr):
    super(GraphRegressionModel, self).__init__()
    self.node_linear =  NodeFeatureLinear(in_node_channels, hidden_channels)
    self.edge_linear = EdgeFeatureLinear(in_edge_channels, hidden_channels)
    self.mpnn_type = mpnn_type
    if mpnn_type == 'gat':
      self.mpnn = tg.nn.GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
    elif mpnn_type == 'gcn':
      self.mpnn = tg.nn.GCNConv(hidden_channels, hidden_channels)
    elif mpnn_type == 'mpnn':
      self.mpnn = MPNN(hidden_channels, hidden_channels, edge_dim=hidden_channels)
    else:
      raise ValueError("Invalid MPNN type. Choose 'gat' or 'gcn' or 'mpnn'.")
    self.aggr = aggr
    self.regression_head = RegressionHead(hidden_channels, out_channels)

  def forward(self, batch, device):
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    edge_attr = batch.edge_attr.to(device)

    x = self.node_linear(x)
    edge_attr = self.edge_linear(edge_attr)
    if self.mpnn_type == 'gcn':
      x = self.mpnn(x, edge_index)
    else:
      x = self.mpnn(x, edge_index, edge_attr)
    x = self.aggr(x, batch.batch.to(device)) # Use batch information for global pooling
    return self.regression_head(x)

class MultiLayerGraphRegressionModel(torch.nn.Module):
  def __init__(self, in_node_channels, in_edge_channels, hidden_channels, out_channels, mpnn_type, aggr, num_layers, dropout_prob=0.1):
    super(MultiLayerGraphRegressionModel, self).__init__()
    self.node_linear =  NodeFeatureLinear(in_node_channels, hidden_channels[0])
    self.edge_linear = EdgeFeatureLinear(in_edge_channels, hidden_channels[0])
    self.mpnn_type = mpnn_type
    self.num_layers = num_layers
    self.mpnns = torch.nn.ModuleList()
    if mpnn_type == 'gat':
      for i in range(num_layers):
        self.mpnns.append(tg.nn.GATConv(hidden_channels[i], hidden_channels[i], edge_dim=hidden_channels[i]))
    elif mpnn_type == 'gcn':
      for i in range(num_layers):
        self.mpnns.append(tg.nn.GCNConv(hidden_channels[i], hidden_channels[i]))
    elif mpnn_type == 'mpnn':
      for i in range(num_layers):
        self.mpnns.append(MPNN(hidden_channels[i], hidden_channels[i], edge_dim=hidden_channels[i]))
    else:
      raise ValueError("Invalid MPNN type. Choose 'gat' or 'gcn' or 'mpnn'.")
    self.node_lin_layers = torch.nn.ModuleList()
    self.edge_lin_layers = torch.nn.ModuleList()
    for i in range(num_layers):
      self.node_lin_layers.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))
      self.edge_lin_layers.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))
    self.aggr = aggr
    self.dropout = torch.nn.Dropout(p=dropout_prob)
    self.regression_head = RegressionHead(hidden_channels[-1], out_channels)

  def forward(self, batch, device):
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    edge_attr = batch.edge_attr.to(device)

    x = self.node_linear(x)
    edge_attr = self.edge_linear(edge_attr)
    
    for i in range(self.num_layers):
      if self.mpnn_type == 'gcn':
        x = self.mpnns[i](x, edge_index)
      else:
        x = self.mpnns[i](x, edge_index, edge_attr)
      x = torch.nn.functional.relu(self.node_lin_layers[i](x))
      edge_attr = torch.nn.functional.relu(self.edge_lin_layers[i](edge_attr))
#      x = self.dropout(x) # Apply dropout after activation
#      edge_attr = self.dropout(edge_attr) # Apply dropout after activation
    
    x = self.aggr(x, batch.batch.to(device)) # Use batch information for global pooling
    return self.regression_head(x)
