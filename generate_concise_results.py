import os
import shutil
import json

def calculate_accuracy(logit_file_path):
    with open(logit_file_path, 'r') as file:
        data = json.load(file)
    
    total_items = len(data)
    correct_items = sum(1 for item in data.values() if item.get("is_correct", False))
    accuracy = correct_items / total_items if total_items > 0 else 0
    
    return accuracy



def copy_concise_results(src, dest):
    # Ensure the destination directory exists
    os.makedirs(dest, exist_ok=True)
    print(f"Destination directory '{dest}' is ready.")

    # Define the subdirectories to copy
    subdirs_to_copy = ['few-sample-ft', 'ft', 'icl', 'detailed-ft']

    for subdir in subdirs_to_copy:
        print(f"Processing subdirectory: {subdir}")
        src_path = os.path.join(src, subdir, 'accuracy_results')
        dest_path = os.path.join(dest, subdir, 'accuracy_results')

        if os.path.exists(src_path):
            print(f"Copying accuracy results from '{src_path}' to '{dest_path}'")
            # Copy the directory structure and files
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path)
            else:
                for item in os.listdir(src_path):
                    s = os.path.join(src_path, item)
                    d = os.path.join(dest_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)

        # Process logits to calculate accuracy
        logits_path = os.path.join(src, subdir, 'logits')
        if os.path.exists(logits_path):
            print(f"Calculating accuracy for logits in '{logits_path}'")
            for root, _, files in os.walk(logits_path):
                for file in files:
                    if file.endswith('logit_data.json'):
                        logit_file_path = os.path.join(root, file)
                        accuracy = calculate_accuracy(logit_file_path)
                        print(f"Calculated accuracy for '{logit_file_path}': {accuracy:.2f}")
                        
                        # Save accuracy result
                        accuracy_result_path = os.path.join(dest, subdir, 'accuracy_results', os.path.relpath(root, logits_path))
                        os.makedirs(accuracy_result_path, exist_ok=True)
                        accuracy_file = os.path.join(accuracy_result_path, 'accuracy.json')
                        with open(accuracy_file, 'w') as acc_file:
                            json.dump({'accuracy': accuracy}, acc_file)
                        print(f"Saved accuracy to '{accuracy_file}'")


    # Copy the entire 'id' directory
    id_src_path = os.path.join(src, 'id')
    id_dest_path = os.path.join(dest, 'id')

    if os.path.exists(id_src_path):
        print(f"Copying 'id' directory from '{id_src_path}' to '{id_dest_path}'")
        if not os.path.exists(id_dest_path):
            shutil.copytree(id_src_path, id_dest_path)
        else:
            for item in os.listdir(id_src_path):
                s = os.path.join(id_src_path, item)
                d = os.path.join(id_dest_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        print(f"Finished copying 'id' directory.")



# Define source and destination directories
source_directory = 'results'
destination_directory = 'concise_results'

# Execute the function
copy_concise_results(source_directory, destination_directory)