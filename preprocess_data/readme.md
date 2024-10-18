# Extract ImageNet VAE Features and Upload to Hugging Face

This guide walks through the process of extracting ImageNet VAE features, merging the files from distributed scripts, and uploading the dataset to Hugging Face.

## 1. Extract ImageNet VAE Features

To extract VAE features from ImageNet and store them into a streaming format, use the following script. You can specify the image size and choose a different VAE model if needed.

```bash
bash preprocess_data/extract_imagefeature_to_streaming.sh
```

You can set parameters such as `image size` and choose a different VAE model to use.

### MosaicML Streaming Documentation:
Refer to the [MosaicML Streaming Docs](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) for further details.

## 2. Merge Extracted Image Features

Since the extraction script (`extract_imagefeature_to_streaming.sh`) is distributed across multiple GPUs, you will need to merge the feature files afterwards. Use the following Python script to merge the files:

```bash
python merge_mds.py
```

### Important:
- Set the path for merging by specifying the `--out_root` parameter. This should match the `--out_root` path used in `preprocess_data/extract_imagefeature_to_streaming.py`.

## 3. Upload Dataset to Hugging Face

Once the dataset is prepared, you can upload it to Hugging Face. Follow the instructions in the [Hugging Face CLI Guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

To upload the current directory as the root of your Hugging Face repo, run the following command:

```bash
huggingface-cli upload wangyanhui666/imagenet_vae_mds_fp32 . --repo-type dataset
```

## 4. Download Dataset from Hugging Face

To download the dataset from Hugging Face, follow these steps:

1. Install the `hf_transfer` package:

```bash
pip install hf_transfer
```

2. Enable the Hugging Face transfer feature:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=True
```

3. Download the dataset from Hugging Face:

```bash
huggingface-cli download wangyanhui666/imagenet_vae_mds_fp32 --repo-type dataset --local-dir ./vae_mds_fp32
```
