from pathlib import Path
from utils import *
from safetensors.torch import load_file
import torch
from tqdm import tqdm

"""this file contains functions to calculate intrinsic dimension"""

def fetch_tensors(path):
    tensors = []
    filenames = []
    for file in path.iterdir():
        if not file.is_file():
            continue
        try:
            tensor = load_file(file)["hidden_state"]
            tensors.append(tensor)
            filenames.append(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if len(tensors) == 0:
        return None
    return torch.stack(tensors).to(dtype=torch.float64), filenames  # assuming all tensors have the same shap





def calculate_lid(data, k=25, random_neighbors=False, verbose=False):
    # Step 1: Compute all pairwise Euclidean distances
    
    if torch.cuda.is_available():
        data = data.to("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
    
    dist_matrix = torch.cdist(data, data, p=2)
    if verbose:
        print("DISTANCE MATRIX")
        print(dist_matrix)
    
    num_points = data.size(0)
    
    if random_neighbors:
        # Step 2a: Randomly select k neighbors for each point
        k_distances = torch.zeros(num_points, k)
        for i in range(num_points):
            # Ensure we don't select the point itself
            possible_indices = torch.cat((torch.arange(i), torch.arange(i+1, num_points)))
            random_indices = possible_indices[torch.randint(0, num_points-1, (k,))]
            k_distances[i] = dist_matrix[i, random_indices]
        # Sort the distances in increasing order
        k_distances, _ = torch.sort(k_distances, dim=1)
    else:
        # Step 2b: Sort each row and get the distances to the k-th nearest neighbors
        # We ignore the first column because it represents the distance of the point to itself (which is 0)
        sorted_distances, _ = torch.sort(dist_matrix, dim=1)
        k_distances = sorted_distances[:, 1:k+1]  # Start from 1 to exclude the distance to itself

    if verbose:
        if random_neighbors:
            print("RANDOM K_DISTANCES")
        else:
            print("SORTED K_DISTANCES")
        print(k_distances.shape)
        print(k_distances)
    
    # Step 3: Calculate LID for each point
    # Use log of ratios of T_k(x) to T_j(x)
    # T_k(x) is repeated across columns for vectorized division
    ratios = k_distances[:, -1:] / k_distances[:, :-1]
    if verbose:
        print("ratios")
        print(ratios)
        print(ratios.shape)
    
    log_ratios = torch.log(ratios)
    mean_log_ratios = torch.mean(log_ratios, dim=1)
    lids = 1 / mean_log_ratios
    lid_list = lids.tolist()

    return lid_list



def calculate_id_two_nn(data, verbose=False):
    # Step 1: Compute all pairwise Euclidean distances
    data = data.cpu()
    dist_matrix = torch.cdist(data, data, p=2)
    if verbose:
        print("DISTANCE MATRIX")
        print(dist_matrix)
    
    num_points = data.size(0)
    
    # Step 2: Find the two nearest neighbors for each point
    sorted_distances, _ = torch.sort(dist_matrix, dim=1)
    r1 = sorted_distances[:, 1]  # First nearest neighbor distance
    r2 = sorted_distances[:, 2]  # Second nearest neighbor distance

    if verbose:
        print("First nearest neighbor distances (r1)")
        print(r1)
        print("Second nearest neighbor distances (r2)")
        print(r2)
    
    # Step 3: Calculate the ratios
    r1[r1 == 0] = 1e-10  # Avoid division by zero
    mu = r2 / r1

    if verbose:
        print("Ratios (mu)")
        print(mu)
    
    # Step 4: Compute empirical CDF
    mu_sorted, _ = torch.sort(mu)
    F_emp = torch.arange(1, num_points + 1, dtype=torch.float32) / num_points

    if verbose:
        print("Sorted ratios (mu_sorted)")
        print(mu_sorted)
        print("Empirical CDF (F_emp)")
        print(F_emp)
    
    # Step 5: Discard top 10% of the highest mu values for stability
    cutoff = int(0.9 * num_points)
    mu_sorted = mu_sorted[:cutoff]
    F_emp = F_emp[:cutoff]

    # Step 6: Logarithmic transformation
    log_mu = torch.log(mu_sorted).to(torch.float32)
    log_one_minus_F = torch.log(1 - F_emp)

    if verbose:
        print("Logarithmic transformation (log_mu)")
        print(log_mu)
        print("Logarithmic transformation (log_one_minus_F)")
        print(log_one_minus_F)
    
    # Step 7: Linear fit to determine slope (intrinsic dimension)
    A = log_mu.unsqueeze(1)  # Shape (N, 1)
    b = log_one_minus_F.unsqueeze(1)

    if verbose:
        print("Matrix A")
        print(A)
        print("Matrix b")
        print(b)
    
    # Solve the least squares problem manually
    ATA = torch.matmul(A.T, A)
    ATb = torch.matmul(A.T, b)
    
    if verbose:
        print("Matrix ATA")
        print(ATA)
        print("Matrix ATb")
        print(ATb)
    
    # Check if ATA is invertible
    try:
        slope = torch.matmul(torch.inverse(ATA), ATb)[0].item()
    except RuntimeError as e:
        if verbose:
            print("Matrix inversion error:", e)
        return float('inf')  # Return infinity to indicate failure

    id = -slope  # The slope gives us the intrinsic dimension

    return id