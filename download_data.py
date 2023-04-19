import json, os, pickle
from tqdm import tqdm
with open("images/det_ingrs.json", "r") as f:
    data = json.load(f)
    
print(len(data))

with open("images/layer1.json", "r") as f:
    dataLayer1 = json.load(f)
    
print(len(dataLayer1))


train_IDs, test_IDs, valid_IDs = [], [], []
layer1Data = []
for k in tqdm(dataLayer1):
    if k["partition"] == "train":
        train_IDs.append(k["id"])
    if k["partition"] == "test":
        test_IDs.append(k["id"])
    if k["partition"] == "val":
        valid_IDs.append(k["id"])
        
    layer1Data.append(k)
        
    if len(layer1Data) == 100000:
        break
    
print(len(valid_IDs), len(train_IDs), len(test_IDs), len(layer1Data))
    
with open("images/layer2.json", "r") as f:
    dataLayer2 = json.load(f)
  
print(len(dataLayer2))



layer2Data = []
for k in tqdm(dataLayer2):
    if k["id"] in train_IDs + test_IDs + valid_IDs:
        layer2Data.append(k)
        
with open('images_250k/layer2Data.pickle', 'wb') as f:
    pickle.dump(layer2Data, f)
with open('images_250k/layer2Data.pickle', 'rb') as f:
    layer2Data = pickle.load(f)
    
import urllib.request, requests
    
import multiprocessing


os.system(f"rm -rf images_250k/val")
os.system(f"rm -rf images_250k/train")
os.system(f"rm -rf images_250k/test")

os.system(f"mkdir images_250k/val/")
os.system(f"mkdir images_250k/train/")
os.system(f"mkdir images_250k/test/")

# def download_images_for_id(args):
#     id_type, id_list = args
#     for k in tqdm(dataLayer2):
#         if k["id"] in id_list:
#             foodId = k["id"]
#             allImages = k["images"]
#             os.system(f"mkdir images_250k/{id_type}/{foodId}")
#             for im in allImages:
#                 downloadImage(im["url"], f"images_250k/{id_type}/{foodId}/{im['id']}")

def download_image(element): 
    output_dir = "train"
    if element["id"] in train_IDs:
        foodId = element["id"]
        allImages = element["images"]
        os.system(f"mkdir images_250k/{output_dir}/{foodId}")
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"images_250k/{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass
    
    output_dir = "test"     
    if element["id"] in test_IDs:
        foodId = element["id"]
        allImages = element["images"]
        os.system(f"mkdir images_250k/{output_dir}/{foodId}")
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"images_250k/{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass
    
    output_dir = "val"         
    if element["id"] in valid_IDs:
        foodId = element["id"]
        allImages = element["images"]
        os.system(f"mkdir images_250k/{output_dir}/{foodId}")
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"images_250k/{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass


print("train started")
def download_images(dataLayer2, num_processes=40):
    with multiprocessing.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(download_image, [(element) for element in dataLayer2])):
            pass
        
download_images(dataLayer2, num_processes=40)
print("download completed")


dataSmall = []
for d in tqdm(data):
    if d["id"] in train_IDs + test_IDs + valid_IDs:
        dataSmall.append(d)
        
print(len(dataSmall))

with open("images_250k/det_ingrs_small.json", "w+") as f:
    json.dump(dataSmall, f)
    
with open("images_250k/layer1_small.json", "w+") as f:
    json.dump(layer1Data, f)
    
with open("images_250k/layer2_small.json", "w+") as f:
    json.dump(layer2Data, f)


