import torch
import torch.nn as nn
import torch.nn.functional as F

'''GNN Layer'''


class LocalGlobalAttentionLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 k_l: float,
                 k_g: float,
                 is_concat: bool = True,
                 dropout: float = 0.5,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.k_l = k_l
        self.k_g = k_g
        self.share_weights = share_weights

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear_score = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.linear_delta = nn.Linear(self.n_hidden * 2, self.n_hidden)

    def forward(self, feats: torch.Tensor, x: torch.Tensor, adj: torch.Tensor):
        n_nodes = x.shape[0]

        g_l = self.linear_l(x).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(x).view(n_nodes, self.n_heads, self.n_hidden)

        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        assert adj.shape[0] == 1 or adj.shape[0] == n_nodes
        assert adj.shape[1] == 1 or adj.shape[1] == n_nodes
        assert adj.shape[2] == 1 or adj.shape[2] == self.n_heads

        # local network (first-order neighbor)
        a_l = self.softmax((e.masked_fill(adj == 0, float('-inf'))))
        a_1nd = a_l * (adj > 0)
        k_local = self.k_l
        keep_l = int(a_1nd.shape[0] * (1 - k_local))
        masks_l = []
        for head in range(a_1nd.shape[2]):
            head_scores = a_1nd[:, :, head]
            _, indices = torch.topk(head_scores, keep_l, dim=0, largest=True)
            head_mask = torch.zeros_like(a_1nd[:, :, head])
            head_mask[indices, :] = 1
            masks_l.append(head_mask)
        a_1nd_filtered = a_1nd * torch.stack(masks_l, dim=2)
        attn_res_local = torch.einsum('ijh,jhf->ihf', a_1nd_filtered, g_r)

        # global network
        omega = self.softmax(e)
        feats_normalized = (feats - feats.min(dim=0).values) / (feats.max(dim=0).values - feats.min(dim=0).values)
        feats_normalized[torch.isnan(feats_normalized)] = 0.0
        g_r_repeat_interleave_normalized = (g_r_repeat_interleave - g_r_repeat_interleave.min(dim=0).values) / (
                g_r_repeat_interleave.max(dim=0).values - g_r_repeat_interleave.min(dim=0).values)
        alpha = self.softmax(
            torch.abs(
                torch.norm(
                    feats_normalized.repeat(n_nodes, 1), p=2, dim=1, keepdim=True
                ) - torch.norm(
                    g_r_repeat_interleave_normalized, dim=2, keepdim=False
                )
            ).view(n_nodes, n_nodes, self.n_heads)
        )
        gamma = 0.5 * (omega + (1 - alpha))
        # global filter
        k_global = self.k_g
        keep_g = int(gamma.shape[0] * (1 - k_global))
        masks_g = []
        for head in range(gamma.shape[2]):
            head_scores_g = gamma[:, :, head]
            _, indices_g = torch.topk(head_scores_g, keep_g, dim=0, largest=True)
            mask_g = torch.zeros_like(gamma[:, :, head])
            mask_g[indices_g, :] = 1
            masks_g.append(mask_g)
        gamma_filtered = gamma * torch.stack(masks_g, dim=2)
        temp_g = 0.001
        gamma_filtered = self.softmax(torch.where(gamma_filtered == 0, torch.tensor(1e-10), gamma_filtered) / temp_g)
        attn_res_global = torch.einsum('ijh,jhf->ihf', gamma_filtered, g_r)

        # information fusion
        interaction = self.activation(self.linear_delta(torch.cat([attn_res_local, attn_res_global], dim=2)))
        delta = self.softmax(interaction)
        attn_res = delta * attn_res_local + (1 - delta) * attn_res_global

        # multi-head mechanism (average)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


'''GNN'''


class GNN(nn.Module):
    def __init__(self,
                 in_features: int,
                 n_hidden_gat: list,
                 n_heads: int,
                 k_l: float,
                 k_g: float,
                 dropout: float,
                 share_weights: bool = True):
        super().__init__()

        self.layer1 = LocalGlobalAttentionLayer(in_features, n_hidden_gat[0], n_heads, k_l, k_g,
                                                is_concat=True, dropout=dropout, share_weights=share_weights)
        self.activation = nn.ELU()
        self.output = LocalGlobalAttentionLayer(n_hidden_gat[0], n_hidden_gat[1], 1, k_l, k_g,
                                                is_concat=False, dropout=dropout, share_weights=share_weights)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor, x: torch.Tensor, adj: torch.Tensor):
        x = self.dropout(x)
        x = self.activation(self.layer1(feats, x, adj))
        x = self.dropout(x)
        x = self.activation(self.output(feats, x, adj))
        return x


'''Classification_layer'''


class NodeClassification(nn.Module):
    def __init__(self, in_features: int,
                 n_hidden: int,
                 n_classes: int):
        super().__init__()

        self.fc1 = nn.Linear(in_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        o = self.softmax(self.fc3(x))
        return o


'''LG-GNN'''


class LG_GNN(nn.Module):

    def __init__(self, in_features: int,
                 n_hidden_gnn: list,
                 n_classes: int,
                 n_heads: int,
                 dropout: float,
                 num_unique_values: int,
                 embed_dim: int,
                 k_l: float,
                 k_g: float,
                 share_weights: bool = True,
                 head='mlp',
                 classifier='linear'):
        super().__init__()

        # embedding layer
        self.embed_dims = embed_dim
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_unique_values, embed_dim).cuda()
            for i in range(in_features)
        ])

        # gnn
        self.gnn = GNN(in_features * embed_dim, n_hidden_gnn, n_heads, k_l, k_g, dropout, share_weights)

        # hidden layers
        if head == 'linear':
            self.head = nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1])
        elif head == 'mlp-2':
            self.head = nn.Sequential(
                nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1]),
                nn.ReLU(inplace=True),
                nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1])
            )
        elif head == 'mlp-3':
            self.head = nn.Sequential(
                nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1]),
                nn.ReLU(inplace=True),
                nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1]),
                nn.ReLU(inplace=True),
                nn.Linear(n_hidden_gnn[1], n_hidden_gnn[1]),
            )

        # classifier
        if classifier == 'mlp':
            self.classifier = NodeClassification(n_hidden_gnn[1], n_hidden_gnn[1] // 2, n_classes)
        elif classifier == 'linear':
            self.classifier = nn.Linear(n_hidden_gnn[1], n_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        embed = []
        # embedding
        for i in range(len(self.embedding_layers)):
            embedded = self.embedding_layers[i](torch.autograd.Variable(x[:, i].long()))
            embed.append(embedded)
        embed = torch.cat(embed, dim=1)
        o_gnn = self.gnn(x, embed, adj)
        o_pro = F.normalize(self.head(o_gnn), dim=1)
        o_c = self.classifier(o_pro)
        return o_c, o_pro
