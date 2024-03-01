# cap6412_assign1
BLIP-2 Part 1

## 0) Requirements

### Python Packages:

Install salesforce-lavis from the original [lavis github repository](https://github.com/salesforce/LAVIS) by cloning it, accessing the cloned repository and perform `pip install -e . ` while inside the repository.

Do not use `pip install salesforce-lavis` as the Blip2OPT model in the PyPI version does not have `predict_answers` for performing VQA.

### Datasets
I have compiled the relevant annotations in [my google driver folder](https://drive.google.com/drive/folders/1lqkK8N5fs0ytKnMetfVOs0v-3tMgESga?usp=sharing), download the `cache` folder and put it in this repository. You should have your annotations in `cache/coco/annotations`, `cache/flickr30k/annotations` and `cache/msvdqa/annotations`.

For COCO images, download the `test2014`, `test2015`, `train2014` and `val2014` folders and store them in `cache/coco/images`. COCO download link: https://cocodataset.org/#download.

For Flickr30k images, download the `flickr30k-images` folder and store it in `cache/flickr30k/images`. Flickr30k download link : https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data

For MSVD videos, download the videos and store it in `cache/msvdqa/videos`. MSVD download link: https://www.cs.utexas.edu/users/ml/clamp/videoDescription/

## 1) Running evaluation
I have compiled all precomputed outputs in [my google driver folder](https://drive.google.com/drive/folders/1lqkK8N5fs0ytKnMetfVOs0v-3tMgESga?usp=sharing), in the 3 folders `precomputed_coco`, `precomputed_flickr` and `precomputed_msvd`. 

There are 5 evaluation scripts in this repository:

- `coco_ret.py` evaluates image-text retrieval on the COCO karpathy-test dataset with a COCO-finetuned BLIP-2 model.
- `flickr_ret.py` evaluates image-text retrieval on the flickr30k test dataset with a COCO-finetuned BLIP-2 model.
- `coco_cap.py` evaluates captioning on the COCO dataset with a COCO-finetuned BLIP-2 OPT6.7b model.
- `coco_vqa.py` evaluates VQA on the COCO dataset with the original BLIP-2 model with OPT6.7b.
- `msvdqa.py` evaluates VideoQA on the MSVD dataset with the original BLIP-2 model with OPT6.7b.

Addendum 1: precomputed outputs are stored in `precomputed_coco`, `precomputed_flickr` and `precomputed_msvd`. If you wish to have BLIP-2 re-generate the outputs (not recommended, it takes hours) in these folders, simply delete all the files inside and run the evaluation scripts. Otherwise, the evaluation scripts will simply use the saved outputs to compute performance metrics.

Addendum 2: inference code in these scripts are ripped from the original [lavis github repository](hhttps://github.com/salesforce/LAVIS) and modified for ease of usage and reading.

## 2) Video Result
A video evidence of running these scripts can be found in the `assets/` folder in this repository.
