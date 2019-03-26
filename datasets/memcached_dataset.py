import mc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img
 
class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, fake=False):
        self.root_dir = root_dir
        self.transform = transform
        self.fake = fake
        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))

        self.initialized = False

        # prepare fake img
        if self.fake:
          img = np.random.randint(0, 256, (350, 350, 3), dtype=np.uint8)
          self.img = Image.fromarray(img)
          if self.transform is not None:
            self.img = self.transform(self.img)
 
    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True
 
    def __getitem__(self, idx):
        if self.fake:
            img = self.img
            cls = 0
        else:
            filename = self.root_dir + '/' + self.metas[idx][0]
            cls = self.metas[idx][1]
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            img = pil_loader(value_str)
        
        ## transform
        if self.transform is not None and not self.fake:
            img = self.transform(img)
        return img, cls
