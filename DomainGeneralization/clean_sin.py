import os

# List of domains and classes
domains = ['Cloud', 'Forest', 'Ocean', 'Origami', 'Sand_dune']
classes = [
    "airplane", "airplane_icon_google", "bicycle", "bird", "bottle", "car", 
    "cat", "dog", "dolphin", "fork", "guitar", "mug", "panda", "paper", 
    "plastic", "sailboat", "scooter", "teapot"
]

# Creating a mapping of class names to indices
class_to_index = {cls: idx for idx, cls in enumerate(classes)}

# Base directory path
root = '/data/SIN'
prompt_type = 'simple_prompts'  # or 'complex_prompts'
difficulty = 'Easy'  # or 'Hard'

# Base path for the output format you want to mimic
output_base = '/data/sin_processed'

# Iterate over each domain
for domain in domains:
    # Open a text file for writing image paths and labels
    with open(f'/data/SIN/{domain.lower()}_images.txt', 'w') as file:
        # Iterate over each class
        for cls in classes:
            # Construct the path for each class based on difficulty level
            class_path = os.path.join(root, prompt_type, cls, difficulty)
            
            # Check if the directory exists
            if os.path.exists(class_path):
                # List all files in the directory
                for filename in os.listdir(class_path):
                    # Check if the file is a PNG image
                    if filename.endswith('.png'):
                        # Construct the full path to be written to the file
                        full_path = os.path.join(class_path, filename)
                        # Get the class index from the mapping
                        class_index = class_to_index[cls]
                        # Write the path and the label to the file
                        file.write(f'{full_path} {class_index}\n')

print("Files have been created for each domain.")
