import os
import tarfile
from tqdm import tqdm

def create_webdataset_shards(features_dir, labels_dir, output_dir, shard_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    
    features_files = sorted(os.listdir(features_dir))
    labels_files = sorted(os.listdir(labels_dir))
    
    assert len(features_files) == len(labels_files), "特征文件和标签文件数量不匹配"
    
    for shard_id in range(0, len(features_files), shard_size):
        shard_features = features_files[shard_id:shard_id + shard_size]
        shard_labels = labels_files[shard_id:shard_id + shard_size]
        shard_filename = os.path.join(output_dir, f"shard-{shard_id // shard_size:06d}.tar")
        
        with tarfile.open(shard_filename, "w") as tar:
            for feature_file, label_file in tqdm(zip(shard_features, shard_labels), total=len(shard_features)):
                feature_path = os.path.join(features_dir, feature_file)
                label_path = os.path.join(labels_dir, label_file)
                
                # 提取文件ID或公共部分作为basename
                feature_basename = os.path.splitext(os.path.basename(feature_file))[0]
                label_basename = os.path.splitext(os.path.basename(label_file))[0]
                
                # 如果文件名不同，例如"feature_1000000.npy"和"label_1000000.npy"
                # 可以提取数字部分
                feature_id = ''.join(filter(str.isdigit, feature_basename))
                label_id = ''.join(filter(str.isdigit, label_basename))
                
                assert feature_id == label_id, "特征文件和标签文件的ID不匹配"
                
                basename = feature_id  # 使用ID作为basename
                
                feature_arcname = f"{basename}.features.npy"
                label_arcname = f"{basename}.labels.npy"
                
                tar.add(feature_path, arcname=feature_arcname)
                tar.add(label_path, arcname=label_arcname)


features_dir = "/home/t2vg-a100-G4-40/t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256/imagenet512_features"
labels_dir = "/home/t2vg-a100-G4-40//t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256/imagenet512_labels"
output_dir = "/home/t2vg-a100-G4-40//t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256/webdataset_shards"
shard_size = 10000

create_webdataset_shards(features_dir, labels_dir, output_dir, shard_size)
