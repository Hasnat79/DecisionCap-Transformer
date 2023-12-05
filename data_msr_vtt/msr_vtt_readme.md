# Instructions for Dataset Generation

## Step 1: Extract Category 15 Caption Data
download msr-vtt train_val_annotation dataset from this [link](https://www.mediafire.com/folder/h14iarbs62e7p/shared) in this folder.
Run the following command in your terminal:

```bash
python cat15_cap_extractor.py
```
## Step 2: Reference Files for CLIP
The following files are reference files for CLIP:

    clip_image_feature.py
    clip_text_embeds.py

## Step 3: [Download](https://drive.google.com/drive/folders/13GfPqJMRMYJFDhvZNZYIB5QN93CXZ03m?usp=drive_link) Cat15_encode_all_frames Directory
Download the Cat15_encode_all_frames directory. This directory was generated in this [kaggle_notebook](https://www.kaggle.com/code/wenyuanli0326/clip-encodings/notebook) and is required for the dataset generation process

## Step 4: Generate Dataset File

Finally, run the following command to generate the dataset file:
```bash
python dataset_builder.py
```
This will create a file named msr_vtt_cat15_d4rl_dataset.json. It will saved on 'gym/data/' folder 

Make sure you have the necessary dependencies installed and proper permissions to run these commands.



