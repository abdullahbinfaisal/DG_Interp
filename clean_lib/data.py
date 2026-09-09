import os
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
pacs_domains = {
    0: "art_painting",
    1: "cartoon",
    2: "photo",
    3: "sketch"
}

# Domain index -> folder name, in the same alphabetical-scan order DomainBed's
# MultipleEnvironmentImageFolder uses (domainbed/datasets.py VLCS/OfficeHome
# ENVIRONMENTS), so env indices in backbone directory names ("..._T3") line up
# with these dicts exactly as they do for PACS.
vlcs_domains = {
    0: "Caltech101",
    1: "LabelMe",
    2: "SUN09",
    3: "VOC2007",
}

officehome_domains = {
    0: "Art",
    1: "Clipart",
    2: "Product",
    3: "Real World",
}

DATASET_DOMAINS = {
    "PACS": pacs_domains,
    "VLCS": vlcs_domains,
    "OfficeHome": officehome_domains,
}

DATASET_ROOTS = {
    "PACS": r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS",
    "VLCS": r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\VLCS",
    "OfficeHome": r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\office_home",
}

# TEST TO TRAIN ENVS
pacs_envs = {
    "T0": [1, 2, 3],
    "T1": [0, 2, 3],
    "T2": [0, 1, 3],
    "T3": [0, 1, 2],
    "T23": [0, 1], 
    "T13": [0, 2], 
    "T12": [0, 3], 
    "T03": [1, 2], 
    "T02": [1, 3], 
    "T01": [2, 3]
} 


class PACSDataset(Dataset):
	def __init__(
		self,
		root_dir: str,
		domains: Optional[List[str]] = None,
		transform=None,
		preload_to_gpu: bool = False,
	) -> None:
		
		if domains is None:
			domains = ["photo", "art_painting", "cartoon", "sketch"]

		self.root_dir = root_dir
		self.domains = domains
		self.transform = transform
		self.preload_to_gpu = preload_to_gpu
		self.device = device

		samples_info = []  # (path, class_name)
		classes = set()

		for domain in domains:
			domain_dir = os.path.join(root_dir, domain)
			for class_entry in sorted(os.scandir(domain_dir), key=lambda e: e.name):
				if not class_entry.is_dir():
					continue
				class_name = class_entry.name
				classes.add(class_name)
				class_dir = os.path.join(domain_dir, class_name)
				for file_name in os.listdir(class_dir):
					if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
						samples_info.append((os.path.join(class_dir, file_name), class_name))

		classes = sorted(classes)
		class_to_idx = {name: idx for idx, name in enumerate(classes)}

		self.preloaded = preload_to_gpu and torch.cuda.is_available()

		if self.preloaded:
			data_tensors = []
			labels = []
			for img_path, class_name in samples_info:
				image = Image.open(img_path).convert("RGB")
				if self.transform is not None:
					tensor = self.transform(image)
				else:
					tensor = transforms.ToTensor()(image)
				data_tensors.append(tensor.to(self.device))
				labels.append(class_to_idx[class_name])

			self.data = torch.stack(data_tensors) if data_tensors else torch.empty(0)
			self.labels = torch.tensor(labels, dtype=torch.long, device=self.device)
			self._length = self.data.shape[0]
		else:
			self.samples = [(path, class_to_idx[class_name]) for path, class_name in samples_info]
			self._length = len(self.samples)

	def __len__(self) -> int:
		return self._length

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		if self.preloaded:
			return self.data[idx], self.labels[idx]

		img_path, label = self.samples[idx]
		image = Image.open(img_path).convert("RGB")
		if self.transform is not None:
			tensor = self.transform(image)
		else:
			tensor = transforms.ToTensor()(image)
		return tensor, torch.tensor(label, dtype=torch.long)


PACS_TRANSFORM = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def Load_PACS_full(
	domains: Optional[List[str]] = None,
	batch_size: int = 64,
	root_dir: str = r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS",
	num_workers: int = 0,
	preload_to_gpu: bool = False,
):
	"""Deterministic loader over *every* image in the requested domains.

	Unlike Load_PACS this applies no train/test split, no shuffling and no
	drop_last, so the same call always yields the same images in the same order
	and nothing is silently discarded.

	Load_PACS returns an 80% shuffled train split with drop_last=True, which
	drops a different tail of images on every call (~7 sketch images, giving
	roughly +/-0.08pp of jitter in reported accuracy). Use this function for
	anything whose number ends up in the paper.

	Returns a single DataLoader, not a pair.
	"""
	if domains is None:
		domains = ["photo", "art_painting", "cartoon", "sketch"]

	dataset = PACSDataset(
		root_dir=root_dir,
		domains=domains,
		transform=PACS_TRANSFORM,
		preload_to_gpu=preload_to_gpu,
	)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=0 if preload_to_gpu else num_workers,
		pin_memory=not preload_to_gpu,
	)


def Load_PACS(
	root_dir: str = r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS",
	domains: Optional[List[str]] = None,
	batch_size: int = 64,
	train_split: float = 0.8,
	seed: int = 42,
	shuffle_train: bool = True,
	drop_last: bool = True,
	preload_to_gpu: bool = False,
	num_workers: int = 0,
):
	if train_split <= 0.0 or train_split >= 1.0:
		raise ValueError("train_split must be in (0, 1)")

	if domains is None:
		domains = ["photo", "art_painting", "cartoon", "sketch"]

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = PACSDataset(
		root_dir=root_dir,
		domains=domains,
		transform=transform,
		preload_to_gpu=preload_to_gpu,
	)

	train_size = int(train_split * len(dataset))
	test_size = len(dataset) - train_size

	# fixed generator for reproducible splits
	g = torch.Generator().manual_seed(seed)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

	if preload_to_gpu and torch.cuda.is_available():
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle_train,
			drop_last=drop_last,
			num_workers=0,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			drop_last=drop_last,
			num_workers=0,
		)
	else:
		pin_memory = True
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle_train,
			drop_last=drop_last,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			drop_last=drop_last,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	return train_loader, test_loader


def Load_Dataset_full(
	dataset: str = "PACS",
	domains: Optional[List[str]] = None,
	batch_size: int = 64,
	root_dir: Optional[str] = None,
	num_workers: int = 0,
	preload_to_gpu: bool = False,
):
	"""Generalisation of Load_PACS_full to any dataset in DATASET_DOMAINS
	(currently PACS, VLCS, OfficeHome). VLCS and OfficeHome share PACS's
	domain/class/*.jpg layout, so PACSDataset is reused unchanged.

	Deterministic: no split, no shuffle, no drop_last. Left as a separate
	function (rather than folding into Load_PACS_full) so the PACS path used
	for every existing score file is untouched.
	"""
	if root_dir is None:
		root_dir = DATASET_ROOTS[dataset]
	if domains is None:
		domains = list(DATASET_DOMAINS[dataset].values())

	pacs_dataset = PACSDataset(
		root_dir=root_dir,
		domains=domains,
		transform=PACS_TRANSFORM,
		preload_to_gpu=preload_to_gpu,
	)

	return DataLoader(
		pacs_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=0 if preload_to_gpu else num_workers,
		pin_memory=not preload_to_gpu,
	)


def Load_Dataset(
	dataset: str = "PACS",
	domains: Optional[List[str]] = None,
	root_dir: Optional[str] = None,
	batch_size: int = 64,
	train_split: float = 0.8,
	seed: int = 42,
	shuffle_train: bool = True,
	drop_last: bool = True,
	preload_to_gpu: bool = False,
	num_workers: int = 0,
):
	"""Generalisation of Load_PACS to any dataset in DATASET_DOMAINS (currently
	PACS, VLCS, OfficeHome). Same 80/20 shuffled-split semantics as Load_PACS —
	this is the SAE-training loader, never the eval loader. Kept separate from
	Load_PACS so the existing PACS training path is untouched.
	"""
	if train_split <= 0.0 or train_split >= 1.0:
		raise ValueError("train_split must be in (0, 1)")

	if root_dir is None:
		root_dir = DATASET_ROOTS[dataset]
	if domains is None:
		domains = list(DATASET_DOMAINS[dataset].values())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	pacs_dataset = PACSDataset(
		root_dir=root_dir,
		domains=domains,
		transform=PACS_TRANSFORM,
		preload_to_gpu=preload_to_gpu,
	)

	train_size = int(train_split * len(pacs_dataset))
	test_size = len(pacs_dataset) - train_size

	g = torch.Generator().manual_seed(seed)
	train_dataset, test_dataset = random_split(pacs_dataset, [train_size, test_size], generator=g)

	num_workers_eff = 0 if preload_to_gpu else num_workers
	pin_memory = not preload_to_gpu

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle_train,
		drop_last=drop_last,
		num_workers=num_workers_eff,
		pin_memory=pin_memory,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=drop_last,
		num_workers=num_workers_eff,
		pin_memory=pin_memory,
	)

	return train_loader, test_loader

