import torch
import io
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset


def pt_loader(path: str, zip_archive: zipfile.ZipFile) -> torch.Tensor:
    with zip_archive.open(path) as file:
        buffer = io.BytesIO(file.read())
        tensor = torch.load(buffer)
        return tensor

class ZipDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str, zipfile.ZipFile], Any],
        zip_file: Union[str, Path],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.zip_file = zip_file
        self.zip_archive = zipfile.ZipFile(self.zip_file, 'r')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(
            self.zip_archive,
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        zip_archive: zipfile.ZipFile,
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")

        instances = []
        directory = Path(directory)
        for zip_info in zip_archive.infolist():
            if zip_info.is_dir():
                continue
            path = Path(zip_info.filename)
            if path.parent.name in class_to_idx:
                if extensions is None or path.suffix in extensions:
                    item = (str(path), class_to_idx[path.parent.name])
                    instances.append(item)
        if not allow_empty and len(instances) == 0:
            raise FileNotFoundError(f"No valid files found in {directory}.")
        return instances

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in Path(directory).glob('*') if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path, self.zip_archive)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

# 创建数据集
dataset = ZipDatasetFolder(
    root='./ILSVRC2012_ldm_512_diffuser1/train',  # 这是 ZIP 文件中的根目录
    loader=pt_loader,
    zip_file='/home/t2vg-a100-G4-40/t2vgusw2/videos/imagenet.zip',
    extensions=('.pt',),
)

# 访问数据集中的样本
for i in range(len(dataset)):
    tensor, label = dataset[i]
    print(f"Sample {i}: Label {label}, Tensor shape: {tensor.shape}")