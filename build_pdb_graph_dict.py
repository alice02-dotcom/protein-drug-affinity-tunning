import os, pickle
import numpy as np
from tqdm import tqdm

def build_pdb_graph_dict(pdb_folder, save_path):
    pdb_graph_dict = {}
    for filename in tqdm(os.listdir(pdb_folder)):
        if not filename.endswith('.npy'):
            continue
        target_id = filename.replace('.npy', '')
        data = np.load(os.path.join(pdb_folder, filename), allow_pickle=True).item()
        pdb_graph_dict[target_id] = data

    with open(save_path, 'wb') as f:
        pickle.dump(pdb_graph_dict, f)
    print(f"Saved to {save_path}")

# 실행 예시
if __name__ == "__main__":
    build_pdb_graph_dict(
        './dataset/KIBA_subset/target/pdb',
        './dataset/KIBA_subset/target/processed/pdb_graph_dict.pkl'
    )