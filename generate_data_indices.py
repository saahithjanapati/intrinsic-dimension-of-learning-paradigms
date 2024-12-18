import random
import json


def generate_icl_indices(num_eval_items=5000, num_train_items=1000, k=10, output_file="indices.json"):
    # Create a dictionary to store the indices for each evaluation item
    random.seed(42)

    indices_dict = {}

    # Loop over each evaluation item
    for i in range(num_eval_items):
        # Generate k unique random indices between 0 and num_train_items - 1
        indices = random.sample(range(num_train_items), k)
        # Store the indices in the dictionary
        indices_dict[i] = indices

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(indices_dict, f, indent=4)

    print(f"Indices saved to {output_file}")
    
    

    
def generate_icl_indices_no_repeat(num_eval_items=5000, num_train_items=1000, k=10, output_file="indices.json"):
    # Create a dictionary to store the indices for each evaluation item
    random.seed(42)

    indices_dict = {}
    total_required_indices = num_eval_items * k

    # Check if we have enough training items to avoid duplicates across all evaluations
    if total_required_indices > num_train_items:
        raise ValueError("Insufficient training items for the required unique indices.")

    # Generate a shuffled list of all indices from the training set
    all_indices = list(range(num_train_items))
    random.shuffle(all_indices)

    # Distribute indices across evaluation items
    for i in range(num_eval_items):
        # Assign k unique indices to each evaluation item
        indices_dict[i] = all_indices[i * k : (i + 1) * k]

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(indices_dict, f, indent=4)

    print(f"Indices saved to {output_file}")
    



for k in range(0, 21):
    generate_icl_indices(num_eval_items=5000, num_train_items=1000, k=k, output_file=f"data_indices/no_repeat_icl_indices_{k}_shot.json")

    
    


def generate_few_sample_ft_indices(num_samples, num_ft_runs=10, num_train_items=1000, output_file="ft_indices.json"):
    # Create a dictionary to store the indices for each evaluation item
    random.seed(42)

    indices_dict = {}
    for i in range(num_ft_runs):
        # Loop over each evaluation item
        # Generate k unique random indices between 0 and num_train_items - 1
        indices = random.sample(range(num_train_items), num_samples)
        # Store the indices in the dictionary
        indices_dict[i] = indices
    
    # Save the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(indices_dict, f, indent=4)


# generate_few_sample_ft_indices(num_samples=5, num_ft_runs=10, num_train_items=1000, output_file="data_indices/ft_indices_5_samples.json")
# generate_few_sample_ft_indices(num_samples=10, num_ft_runs=10, num_train_items=1000, output_file="data_indices/ft_indices_10_samples.json") 