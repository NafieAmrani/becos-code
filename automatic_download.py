import requests
from tqdm import tqdm
import zipfile
import os


def download_dataset(zip_path, output_dir, datset_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    response = requests.get(zip_path, stream=True)
    if response.status_code == 200:
        zip_path = os.path.join(output_dir, "kids_dataset.zip")
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {datset_name} dataset"):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        os.remove(zip_path)
        print(f"{datset_name} dataset downloaded and extracted successfully.")
    else:
        print(f"Failed to download {datset_name} dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    zip_path = "https://cvg.cit.tum.de/_media/spezial/bib/kids.zip"
    output_directory = "./data/KIDS/"
    download_dataset(zip_path, output_directory, datset_name="KIDS")
    zip_path = "http://robertodyke.com/shrec2020/SHREC20b_hires.zip"
    output_directory = "./data/Shrec20/"
    download_dataset(zip_path, output_directory, datset_name="Shrec20")