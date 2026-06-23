import torch.nn as nn
import torch
class PCEmbedding(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
class TextPCMatrixEmbedding(nn.Module):
    def __init__(self,args, pc_embed_dim: int = 768, matrix_embed_dim: int = 256, text_embed_dim: int = 768):
        super().__init__()
        self.pc_proj = nn.Linear(pc_embed_dim, text_embed_dim)
        self.matrix_proj = nn.Linear(matrix_embed_dim, text_embed_dim)
        #self.text_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.fusion = args.fusion
    def forward(self, pc_embeds: torch.FloatTensor, matrix_embeds: torch.FloatTensor):
        text_pc_embeds = self.pc_proj(pc_embeds)
        text_matrix_embeds = self.matrix_proj(matrix_embeds)
        #text_pc_embeds = pc_embeds
        #text_matrix_embeds = matrix_embeds
        if self.fusion=='add':
            return text_pc_embeds + text_matrix_embeds
        elif self.fusion=='sq_cat':
            return torch.cat([text_pc_embeds.unsqueeze(1), text_matrix_embeds.unsqueeze(1)],dim=1)
        else:
            raise NotImplementedError("error fusion.")
