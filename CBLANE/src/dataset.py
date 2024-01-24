import gdown
import numpy as np

drive_urls = [
              "https://drive.google.com/uc?id=1VpEFrsJp55exyqpbasg030bdWsCjEVvo",
              "https://drive.google.com/uc?id=1--Z1W9jc-pCfSk111HEZeV0xP4RT6AFA",
              "https://drive.google.com/uc?id=1-77ZWy7PUcFzfRDnRi7VGCfgP0o-Pniw",
              "https://drive.google.com/uc?id=1-9yvd5ILQoNx41HxFZXR3pCfZTVe4or4",
              "https://drive.google.com/uc?id=1-E5JeeXU3c7ZxeUm4VNSoyiwi5IrA2Av",
              "https://drive.google.com/uc?id=1-EVSlHK45JIU47yFCBYjd47FPAKMh6XK",
              "https://drive.google.com/uc?id=1CMplHD3owy0VKbr4MdnfZZVuZhT5iYfW",
              "https://drive.google.com/uc?id=1XGijjStzbbrSRHHrhlw4Pp8pk_xQ-F8k",
              "https://drive.google.com/uc?id=1iuHDEQz-JC5JRbwLJCBogoESIk2zU1MH"
              ]

dataset = [
            "test.npz",
            "validation.npz",
            "train.npz",
            "encoded_train.npz",
            "encoded_test.npz",
            "encoded_validation.npz",
            "690chip.zip",
            "medium_dataset.npz",
            "large_dataset.npz"
          ]

for data,drive_url in zip(dataset,drive_urls):
  gdown.download(drive_url,data,quiet=False)
  
def save_or_load_numpy(option,file,labels=None,sequences=None):
  if option=="save":
    np.savez(file,labels = labels,sequences = sequences)
    return None,None
  if option == "load":
    loaded_array = np.load(file)
    sequences = loaded_array['sequences']
    labels = loaded_array['labels']
    return sequences,labels