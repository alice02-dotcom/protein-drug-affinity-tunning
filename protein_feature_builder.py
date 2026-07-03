import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

# ---------------------------
# AAindex 기반 키 목록
# ---------------------------
AAINDEX_KEYS = [
    'HOPT810101', 'ZIMJ680101', 'ZIMJ680104', 'GRAR740102',
    'FAUJ880103', 'BHAR880101', 'JANJ780101', 'CHOP780201'
]

# ---------------------------
# AAindex 수치 사전
# ---------------------------
AAINDEX_DICT = {
    'HOPT810101': {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
                   'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5},
    'ZIMJ680101': {'A': 0.83, 'R': 0.83, 'N': 0.09, 'D': 0.64, 'C': 1.48, 'Q': 0.0, 'E': 0.65, 'G': 0.1, 'H': 1.1, 'I': 3.07,
                   'L': 2.52, 'K': 1.6, 'M': 1.4, 'F': 2.75, 'P': 2.7, 'S': 0.14, 'T': 0.54, 'W': 0.31, 'Y': 2.97, 'V': 1.79},
    'ZIMJ680104': {'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.05, 'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
                   'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68, 'T': 5.66, 'W': 5.89, 'Y': 5.66, 'V': 5.96},
    'GRAR740102': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
                   'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9},
    'FAUJ880103': {'A': 1.00, 'R': 6.13, 'N': 2.95, 'D': 2.78, 'C': 2.43, 'Q': 3.95, 'E': 3.78, 'G': 0.00, 'H': 4.66, 'I': 4.00,
                   'L': 4.00, 'K': 4.77, 'M': 4.43, 'F': 5.89, 'P': 2.72, 'S': 1.60, 'T': 2.60, 'W': 8.08, 'Y': 6.47, 'V': 3.00},
    'BHAR880101': {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462,
                   'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386},
    'JANJ780101': {'A': 27.8, 'R': 94.7, 'N': 60.1, 'D': 60.6, 'C': 15.5, 'Q': 68.7, 'E': 68.2, 'G': 24.5, 'H': 50.7, 'I': 22.8,
                   'L': 27.6, 'K': 103.0, 'M': 33.5, 'F': 25.5, 'P': 51.5, 'S': 42.0, 'T': 45.0, 'W': 34.7, 'Y': 55.2, 'V': 23.7},
    'CHOP780201': {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70, 'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
                   'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06}
}

#--------------------------------------------------------------------
#   AAindex 기반 아미노산 임베딩 생성기
#   - 각 아미노산을 선택된 AAindex 특성 키에 따라 실수 벡터로 변환
#--------------------------------------------------------------------
class AAindexFeatureEmbedder(nn.Module):

    def __init__(self, aaindex_dict, selected_keys):
        super().__init__()
        self.aaindex_dict = aaindex_dict
        self.selected_keys = selected_keys

    def forward(self, sequence):
        features = []
        for aa in sequence:
            vec = [self.aaindex_dict[k].get(aa, 0.0) for k in self.selected_keys]
            features.append(vec)
        return torch.tensor(features, dtype=torch.float)
    
#--------------------------------------------------------------------
#   AAindex feature 정규화 (z-score 기준)
#   - feature별 평균 0, 표준편차 1로 맞춰서 학습 안정화
#--------------------------------------------------------------------
def normalize_features(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    return (x - mean) / (std + eps)

#--------------------------------------------------------------------
#   구조 기반 피처(fingerprint)와 AAindex 기반 물리화학 피처를 concat
#   - 길이가 다를 경우 앞에서 맞춰 자름 (min length 기준)
#--------------------------------------------------------------------
def concat_protein_features(structure_feat: torch.Tensor, aaindex_feat: torch.Tensor) -> torch.Tensor:
    min_len = min(structure_feat.shape[0], aaindex_feat.shape[0])
    return torch.cat([structure_feat[:min_len], aaindex_feat[:min_len]], dim=1)

#--------------------------------------------------------------------
#   전체 프로세스를 통합하여 protein representation(x)을 생성
#   Args:
#       sequence (str): 단백질 서열
#       fingerprint (np.ndarray or Tensor): 구조 기반 피처
#       embedder (AAindexFeatureEmbedder): AAindex 임베더 객체
#       normalize (bool): 정규화 여부
#   Returns:
#       x (Tensor): (L, D+8) 형태의 노드 피처 텐서
#--------------------------------------------------------------------
def build_protein_representation(sequence, fingerprint, embedder=None, normalize=True):

    if embedder is None:
        raise ValueError("AAindexFeatureEmbedder를 먼저 생성해야 합니다.")

    aaindex_feat = embedder(sequence)
    if normalize:
        aaindex_feat = normalize_features(aaindex_feat)

    if not isinstance(fingerprint, torch.Tensor):
        fingerprint = torch.tensor(fingerprint, dtype=torch.float)

    return concat_protein_features(fingerprint, aaindex_feat)

class ProteinBranch(nn.Module):
    #구조 기반 GNN과 AAindex 기반 서열 임베딩 차원 설정
    def __init__(self, gnn_structure, aaindex_dim):
        super().__init__()
        self.gnn_structure = gnn_structure
        self.aaindex_dim = aaindex_dim

    def forward(self, graph_batch, sequence_tensor):
        # graph_batch: PyG Batch 객체 (x, edge_index, edge_attr, batch)
        # sequence_tensor: (B, L, aaindex_dim) 형태의 단백질 서열 AAindex 임베딩

        x, edge_index, edge_attr, batch_vec = (
            graph_batch.x, graph_batch.edge_index,
            graph_batch.edge_attr, graph_batch.batch
        )
        #단백질 구조(GNN)와 서열(AAindex) 임베딩
        gnn_out = self.gnn_structure(x, edge_index, edge_attr, batch_vec)
        graph_repr = global_mean_pool(gnn_out, batch_vec)  # (B, hidden_dim)

        # sequence_tensor는 (B, L, D) → 평균 Pool
        seq_repr = sequence_tensor.mean(dim=1)  # (B, D)

        #두 임베딩 차원을 합산하여 반환
        return torch.cat([graph_repr, seq_repr], dim=-1)

    @property
    def output_dim(self):
        return self.gnn_structure.hidden_channels + self.aaindex_dim