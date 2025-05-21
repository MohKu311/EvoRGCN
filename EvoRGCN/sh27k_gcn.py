import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the GCN model
class GCNLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPrediction, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

# Function to preprocess a dataframe and train the model
def process_dataframe(df, mode_name):
    print(f"Processing {mode_name} dataset...")
    
    # Step 1: Create the graph
    protein_ids = pd.concat([df['item_id_a'], df['item_id_b']]).unique()
    protein_to_idx = {protein: idx for idx, protein in enumerate(protein_ids)}
    
    edge_index = np.array([
        [protein_to_idx[protein] for protein in df['item_id_a']],
        [protein_to_idx[protein] for protein in df['item_id_b']]
    ])
    
    edge_weights = df['score'].values
    
    # Step 2: Create node features (one-hot encoding of sequences)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    encoder = OneHotEncoder(categories=[amino_acids], sparse_output=False)
    
    node_features = []
    for protein in protein_ids:
        sequence_a = df[df['item_id_a'] == protein]['sequence_a']
        sequence_b = df[df['item_id_b'] == protein]['sequence_b']
        
        if len(sequence_a) > 0:
            sequence = sequence_a.values[0]
        elif len(sequence_b) > 0:
            sequence = sequence_b.values[0]
        else:
            raise ValueError(f"Protein {protein} not found in either 'item_id_a' or 'item_id_b'.")
        
        encoded_seq = encoder.fit_transform(np.array(list(sequence)).reshape(-1, 1))
        node_features.append(encoded_seq.mean(axis=0))
    
    node_features_array = np.array(node_features)
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(node_features_array, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float)  # Optional
    )
    
    # Step 3: Split the data
    edge_index = data.edge_index.numpy()
    edge_index_train, edge_index_test = train_test_split(edge_index.T, test_size=0.2, random_state=42)
    edge_index_train, edge_index_val = train_test_split(edge_index_train, test_size=0.1, random_state=42)
    
    edge_index_train = torch.tensor(edge_index_train, dtype=torch.long).t()
    edge_index_val = torch.tensor(edge_index_val, dtype=torch.long).t()
    edge_index_test = torch.tensor(edge_index_test, dtype=torch.long).t()
    
    # Step 4: Generate negative samples
    neg_edge_index_train = negative_sampling(edge_index_train, num_nodes=data.num_nodes)
    neg_edge_index_val = negative_sampling(edge_index_val, num_nodes=data.num_nodes)
    neg_edge_index_test = negative_sampling(edge_index_test, num_nodes=data.num_nodes)
    
    # Step 5: Train the model
    model = GCNLinkPrediction(in_channels=data.num_features, hidden_channels=16, out_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    def train():
        model.train()
        optimizer.zero_grad()
        
        pos_edge_index = edge_index_train
        neg_edge_index = neg_edge_index_train
        
        z = model(data.x, pos_edge_index)
        pos_score = model.decode(z, pos_edge_index)
        neg_score = model.decode(z, neg_edge_index)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones(pos_score.size(0))))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros(neg_score.size(0))))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    for epoch in range(1, 201):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # Step 6: Evaluate the model
    def evaluate(edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model(data.x, edge_index)
            pos_score = model.decode(z, edge_index).sigmoid().cpu().numpy()
            neg_score = model.decode(z, neg_edge_index).sigmoid().cpu().numpy()
            
            y_true = np.hstack([np.ones(pos_score.size), np.zeros(neg_score.size)])
            y_score = np.hstack([pos_score, neg_score])
            
            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)
            
            return auc_roc, auc_pr
    
    val_auc_roc, val_auc_pr = evaluate(edge_index_val, neg_edge_index_val)
    test_auc_roc, test_auc_pr = evaluate(edge_index_test, neg_edge_index_test)
    
    print(f"{mode_name} - Validation AUC-ROC: {val_auc_roc:.4f}, Validation AUC-PR: {val_auc_pr:.4f}")
    print(f"{mode_name} - Test AUC-ROC: {test_auc_roc:.4f}, Test AUC-PR: {test_auc_pr:.4f}")
    
    # Step 7: Save the model
    torch.save(model.state_dict(), f"gcn_link_prediction_{mode_name}.pth")
    
    # Step 8: Visualize node embeddings
    z = model.encode(data.x, data.edge_index).detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10)
    plt.title(f"t-SNE Visualization of Node Embeddings ({mode_name})")
    plt.savefig(f"node_embeddings_{mode_name}.png")
    plt.show()

# Load CSV files into DataFrames
file_names = {
    "activation": "activation.csv",
    "binding": "binding.csv",
    "catalysis": "catalysis.csv",
    "expression": "expression.csv",
    "inhibition": "inhibition.csv",
    "ptmod": "ptmod.csv",
    "reaction": "reaction.csv"
}

df_activation = pd.read_csv(file_names["activation"])
df_binding = pd.read_csv(file_names["binding"])
df_catalysis = pd.read_csv(file_names["catalysis"])
df_expression = pd.read_csv(file_names["expression"])
df_inhibition = pd.read_csv(file_names["inhibition"])
df_ptmod = pd.read_csv(file_names["ptmod"])
df_reaction = pd.read_csv(file_names["reaction"])

# Process each dataframe
dataframes = {
    "activation": df_activation,
    "binding": df_binding,
    "catalysis": df_catalysis,
    "expression": df_expression,
    "inhibition": df_inhibition,
    "ptmod": df_ptmod,
    "reaction": df_reaction
}

for mode_name, df in dataframes.items():
    process_dataframe(df, mode_name)