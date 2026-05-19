import shutil
import os
import kagglehub

path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")

destination = "./data/yelp"

os.makedirs(destination, exist_ok=True)

for file in os.listdir(path):
    shutil.copy(
        os.path.join(path, file),
        destination
    )

print("Dataset copied to:", destination)