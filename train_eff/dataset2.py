import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _to_unix(path: str) -> str:
    return path.replace("\\", "/").strip()


def _list_view_images(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"View directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"View directory is not a directory: {root}")

    image_map: Dict[str, Path] = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            path = Path(dp) / fn
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = _to_unix(str(path.relative_to(root)))
            image_map[rel] = path
    if not image_map:
        raise RuntimeError(f"No images found under: {root}")
    return image_map


def _build_train_pairs(raw_root: Path, detail_root: Path, class_to_idx: Dict[str, int]):
    raw_map = _list_view_images(raw_root)
    detail_map = _list_view_images(detail_root)

    raw_keys = set(raw_map.keys())
    detail_keys = set(detail_map.keys())
    missing_detail = sorted(raw_keys - detail_keys)
    missing_raw = sorted(detail_keys - raw_keys)
    if missing_detail:
        raise RuntimeError(f"Detail view is missing {len(missing_detail)} files. First missing pair: {missing_detail[0]}")
    if missing_raw:
        raise RuntimeError(f"Raw view is missing {len(missing_raw)} files. First extra detail pair: {missing_raw[0]}")

    samples = []
    for rel in sorted(raw_map.keys()):
        class_name = Path(rel).parts[0]
        if class_name not in class_to_idx:
            raise RuntimeError(f"Class '{class_name}' from raw view not found in CLASSNAME_FILE")
        samples.append((raw_map[rel], detail_map[rel], class_to_idx[class_name]))
    return samples


def _build_single_view_samples(view_root: Path, class_to_idx: Dict[str, int]):
    image_map = _list_view_images(view_root)
    samples = []
    for rel in sorted(image_map.keys()):
        class_name = Path(rel).parts[0]
        if class_name not in class_to_idx:
            raise RuntimeError(f"Class '{class_name}' from view not found in CLASSNAME_FILE")
        samples.append((image_map[rel], class_to_idx[class_name]))
    return samples


def _build_inference_pairs(raw_root: Path, detail_root: Path):
    raw_map = _list_view_images(raw_root)
    detail_map = _list_view_images(detail_root)

    raw_keys = set(raw_map.keys())
    detail_keys = set(detail_map.keys())
    missing_detail = sorted(raw_keys - detail_keys)
    missing_raw = sorted(detail_keys - raw_keys)
    if missing_detail:
        raise RuntimeError(f"Detail test view is missing {len(missing_detail)} files. First missing pair: {missing_detail[0]}")
    if missing_raw:
        raise RuntimeError(f"Raw test view is missing {len(missing_raw)} files. First extra detail pair: {missing_raw[0]}")

    return [(rel, raw_map[rel], detail_map[rel]) for rel in sorted(raw_map.keys())]


class DualViewDataset(Dataset):
    def __init__(self, raw_root: str, detail_root: str, class_to_idx: Dict[str, int], transform=None):
        self.raw_root = Path(raw_root)
        self.detail_root = Path(detail_root)
        self.transform = transform
        self.samples = _build_train_pairs(self.raw_root, self.detail_root, class_to_idx)
        self.labels = [label for _, _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, detail_path, label = self.samples[idx]
        raw_image = Image.open(raw_path).convert("RGB")
        detail_image = Image.open(detail_path).convert("RGB")
        if self.transform:
            raw_image = self.transform(raw_image)
            detail_image = self.transform(detail_image)
        return raw_image, detail_image, label, str(raw_path), str(detail_path)


class DualViewSubsetDataset(Dataset):
    def __init__(self, dataset: DualViewDataset, indices: Sequence[int], transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.labels = [self.dataset.labels[idx] for idx in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_path, detail_path, label = self.dataset.samples[self.indices[idx]]
        raw_image = Image.open(raw_path).convert("RGB")
        detail_image = Image.open(detail_path).convert("RGB")
        if self.transform:
            raw_image = self.transform(raw_image)
            detail_image = self.transform(detail_image)
        return raw_image, detail_image, label, str(raw_path), str(detail_path)


class DualViewInferenceDataset(Dataset):
    def __init__(self, raw_root: str, detail_root: str, transform):
        self.raw_root = Path(raw_root)
        self.detail_root = Path(detail_root)
        self.transform = transform
        self.samples = _build_inference_pairs(self.raw_root, self.detail_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, raw_path, detail_path = self.samples[idx]
        raw_image = Image.open(raw_path).convert("RGB")
        detail_image = Image.open(detail_path).convert("RGB")
        raw_image = self.transform(raw_image)
        detail_image = self.transform(detail_image)
        return raw_image, detail_image, rel_path, str(raw_path), str(detail_path)


class SingleViewInferenceDataset(Dataset):
    def __init__(self, view_root: str, transform):
        self.view_root = Path(view_root)
        self.transform = transform
        image_map = _list_view_images(self.view_root)
        self.samples = [(rel, image_map[rel]) for rel in sorted(image_map.keys())]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, rel_path, str(image_path)


class SingleViewPseudoDataset(Dataset):
    def __init__(self, pseudo_samples: Sequence[Dict[str, object]], transform):
        self.samples = list(pseudo_samples)
        self.transform = transform
        self.labels = [int(sample["pred_label"]) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = str(sample["image_path"])
        label = int(sample["pred_label"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label, image_path


class SingleViewDataset(Dataset):
    def __init__(self, view_root: str, class_to_idx: Dict[str, int], transform=None):
        self.view_root = Path(view_root)
        self.transform = transform
        self.samples = _build_single_view_samples(self.view_root, class_to_idx)
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(image_path)


class SingleViewSubsetDataset(Dataset):
    def __init__(self, dataset: SingleViewDataset, indices: Sequence[int], transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.labels = [self.dataset.labels[idx] for idx in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(image_path)


def _build_basic_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def _split_indices_stratified(labels: Sequence[int], val_ratio: float, seed: int):
    if val_ratio <= 0:
        return list(range(len(labels))), []

    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for _, indices in sorted(class_to_indices.items()):
        indices = indices.copy()
        rng.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
    return train_indices, val_indices


def _build_loader_kwargs(config):
    num_workers = int(getattr(config, "NUM_WORKERS", 4))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(config, "PIN_MEMORY", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(config, "PERSISTENT_WORKERS", True))
        loader_kwargs["prefetch_factor"] = int(getattr(config, "PREFETCH_FACTOR", 2))
    return loader_kwargs


def get_dualview_dataloaders(config):
    with open(config.CLASSNAME_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    full_dataset = DualViewDataset(
        raw_root=config.RAW_TRAIN_DIR,
        detail_root=config.DETAIL_TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=None,
    )

    train_indices, val_indices = _split_indices_stratified(
        full_dataset.labels,
        val_ratio=float(getattr(config, "VAL_SPLIT", 0.2)),
        seed=int(getattr(config, "SEED", 17)),
    )

    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    train_dataset = DualViewSubsetDataset(full_dataset, train_indices, transform=transform)
    val_dataset = DualViewSubsetDataset(full_dataset, val_indices, transform=transform) if val_indices else None

    loader_kwargs = _build_loader_kwargs(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 32)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(getattr(config, "BATCH_SIZE", 32)),
            shuffle=False,
            **loader_kwargs,
        )
    return train_loader, val_loader, class_names


def get_train2_dataloaders(config):
    train_mode = getattr(config, "TRAIN_MODE", "dual_view")
    if train_mode == "dual_view":
        return get_dualview_dataloaders(config)

    if train_mode != "single_view":
        raise ValueError(f"Unknown TRAIN_MODE: {train_mode}")

    with open(config.CLASSNAME_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    single_view_source = getattr(config, "SINGLE_VIEW_SOURCE", "raw")
    if single_view_source == "raw":
        view_root = config.RAW_TRAIN_DIR
    elif single_view_source == "detail":
        view_root = config.DETAIL_TRAIN_DIR
    else:
        raise ValueError(f"Unknown SINGLE_VIEW_SOURCE: {single_view_source}")

    full_dataset = SingleViewDataset(
        view_root=view_root,
        class_to_idx=class_to_idx,
        transform=None,
    )

    train_indices, val_indices = _split_indices_stratified(
        full_dataset.labels,
        val_ratio=float(getattr(config, "VAL_SPLIT", 0.2)),
        seed=int(getattr(config, "SEED", 17)),
    )

    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    train_dataset = SingleViewSubsetDataset(full_dataset, train_indices, transform=transform)
    val_dataset = SingleViewSubsetDataset(full_dataset, val_indices, transform=transform) if val_indices else None

    loader_kwargs = _build_loader_kwargs(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 32)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(getattr(config, "BATCH_SIZE", 32)),
            shuffle=False,
            **loader_kwargs,
        )
    return train_loader, val_loader, class_names


def get_singleview_train2_dataloaders(config):
    with open(config.CLASSNAME_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    single_view_source = getattr(config, "SINGLE_VIEW_SOURCE", "raw")
    if single_view_source == "raw":
        view_root = config.RAW_TRAIN_DIR
    elif single_view_source == "detail":
        view_root = config.DETAIL_TRAIN_DIR
    else:
        raise ValueError(f"Unknown SINGLE_VIEW_SOURCE: {single_view_source}")

    full_dataset = SingleViewDataset(
        view_root=view_root,
        class_to_idx=class_to_idx,
        transform=None,
    )

    train_indices, val_indices = _split_indices_stratified(
        full_dataset.labels,
        val_ratio=float(getattr(config, "VAL_SPLIT", 0.2)),
        seed=int(getattr(config, "SEED", 17)),
    )

    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    train_dataset = SingleViewSubsetDataset(full_dataset, train_indices, transform=transform)
    clean_val_dataset = SingleViewSubsetDataset(full_dataset, val_indices, transform=transform) if val_indices else None
    defect_val_dataset = SingleViewSubsetDataset(full_dataset, val_indices, transform=transform) if val_indices else None

    loader_kwargs = _build_loader_kwargs(config)
    batch_size = int(getattr(config, "BATCH_SIZE", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    clean_val_loader = None
    defect_val_loader = None
    if clean_val_dataset is not None:
        clean_val_loader = DataLoader(
            clean_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        defect_val_loader = DataLoader(
            defect_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    return train_loader, clean_val_loader, defect_val_loader, class_names


def get_dualview_test_dataloader(config):
    with open(config.CLASSNAME_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    dataset = DualViewInferenceDataset(
        raw_root=config.RAW_TEST_DIR,
        detail_root=config.DETAIL_TEST_DIR,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 64)),
        shuffle=False,
        **_build_loader_kwargs(config),
    )
    return loader, class_names


def get_singleview_test_dataloader(config):
    with open(config.CLASSNAME_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    single_view_source = getattr(config, "SINGLE_VIEW_SOURCE", "raw")
    if single_view_source == "raw":
        view_root = config.RAW_TEST_DIR
    elif single_view_source == "detail":
        view_root = config.DETAIL_TEST_DIR
    else:
        raise ValueError(f"Unknown SINGLE_VIEW_SOURCE: {single_view_source}")

    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    dataset = SingleViewInferenceDataset(
        view_root=view_root,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 64)),
        shuffle=False,
        **_build_loader_kwargs(config),
    )
    return loader, class_names


def get_singleview_unlabeled_dataloader(config, view_root: str = None):
    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    if view_root is None:
        view_root = getattr(config, "SELF_TRAIN_SOURCE_DIR")
    dataset = SingleViewInferenceDataset(
        view_root=view_root,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 64)),
        shuffle=False,
        **_build_loader_kwargs(config),
    )
    return loader


def get_singleview_pseudo_dataloader(config, pseudo_samples: Sequence[Dict[str, object]]):
    transform = _build_basic_transform(int(getattr(config, "IMAGE_SIZE", 224)))
    dataset = SingleViewPseudoDataset(
        pseudo_samples=pseudo_samples,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 32)),
        shuffle=True,
        **_build_loader_kwargs(config),
    )
    return loader
