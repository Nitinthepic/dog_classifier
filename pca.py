import os
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from multiprocessing import Pool

def _pca_image(image, image_folder, folder, file_name):
     print(f"Running: {file_name}")
     with Image.open(image) as im:
        pixels = np.array(im)
        r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        pca_r = PCA(n_components=0.99)
        pca_g = PCA(n_components=0.99)
        pca_b = PCA(n_components=0.99)
        pca_r_trans = pca_r.fit_transform(r)
        pca_g_trans = pca_g.fit_transform(g)
        pca_b_trans = pca_b.fit_transform(b)
        pca_r_org = pca_r.inverse_transform(pca_r_trans)
        pca_g_org = pca_g.inverse_transform(pca_g_trans)
        pca_b_org = pca_b.inverse_transform(pca_b_trans)
                 
        temp = np.dstack((pca_r_org,pca_g_org,pca_b_org))
        temp = temp.astype(np.uint8)
        new_image = Image.fromarray(temp)
        new_image.convert("RGB")

        if not os.path.exists(f"{image_folder}/{folder}"):
            os.mkdir(f"{image_folder}/{folder}")
        new_image.save(f"{image_folder}/{folder}/{file_name}")
        print(f"Completed: {file_name}")


def dog_pca(source_data_path, image_path):
        all_images = glob.glob(f"{source_data_path}/**/*.jpg", recursive=True)
        args = list()
        for image in all_images:
            folder = image.split("/")[2]
            file_name = image.split("/")[3]
            args.append((image, image_path, folder, file_name))

        with Pool() as pool:
            pool.starmap(_pca_image, args)

        
            