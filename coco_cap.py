import torch
from PIL import Image
import numpy as np
import os
import json

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from datasets import load_dataset

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

device='cuda'

################
# Load Dataset #
################
YOUR_LOCAL_PATH = 'cache/coco'
coco_test = load_dataset(YOUR_LOCAL_PATH, data_files={'test':'annotations/coco_karpathy_test.json'})['test']

#############################
# Load Model and Processors #
#############################
if 'cap_result.json' not in os.listdir('precomputed_coco'):
	model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device)

#######################################
# Preprocess data for captioning task #
#######################################
image = []

for img_id, ann in enumerate(coco_test):
	image.append(ann["image"])

#########################################################
# Generate a caption for all images, then store in json #
#########################################################
results = []
if 'cap_result.json' not in os.listdir('precomputed_coco'):
	for i in range(len(image)):
		print('i:',i)
		raw_image = Image.open(YOUR_LOCAL_PATH+'/images/'+image[i]).convert("RGB")
		image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		cap = model.generate({"image": image_processed})[0]
		img_id = int(image[i][-16:-4].strip('0'))
		results.append({'image_id': img_id, 'caption': cap})
	with open('precomputed_coco/cap_result.json', 'w') as f:
		json.dump(results, f)
		
###############################
# CIDER and SPICE computation #
###############################
YOUR_LOCAL_PATH = 'cache/coco'
annotation_file = YOUR_LOCAL_PATH + '/annotations/coco_karpathy_test_gt.json'
results_file = 'precomputed_coco/cap_result.json'

# generate new gt based on results
with open('precomputed_coco/cap_result.json','rb') as f:
	res = json.load(f)
	annsImgIds = [ann['image_id'] for ann in res]

with open(YOUR_LOCAL_PATH + '/annotations/coco_karpathy_test_gt.json','rb') as f:
	gt = json.load(f)

# filter results here
gtImgIds = list(map(lambda x: x['id'], gt['images']))

overlap = set(annsImgIds) & set(gtImgIds)

for annid in annsImgIds:
	if annid not in overlap:
		#remove from results
		for ele in res:
			if ele['image_id'] == annid:
				res.remove(ele)
		#remove from gt images
		for ele in gt['images']:
			if ele['id'] == annid:
				gt['images'].remove(ele)
		#remove from gt annotations
		for ele in gt['annotations']:
			if ele['id'] == annid or ele['image_id'] == annid:
				gt['annotations'].remove(ele)
				
for annid in gtImgIds:
	if annid not in overlap:
		#remove from results
		for ele in res:
			if ele['image_id'] == annid:
				res.remove(ele)
		#remove from gt images
		for ele in gt['images']:
			if ele['id'] == annid:
				gt['images'].remove(ele)
		#remove from gt annotations
		for ele in gt['annotations']:
			if ele['id'] == annid or ele['image_id'] == annid:
				gt['annotations'].remove(ele)

with open(YOUR_LOCAL_PATH + '/annotations/filtered_test_gt.json', 'w') as f:
	json.dump(gt, f)				
with open('precomputed_coco/filtered_cap_result.json', 'w') as f:
	json.dump(res, f)

# filtered gt analysis
fgtImgIds = []
for ele in gt['annotations']:
	if ele['image_id'] not in fgtImgIds:
		fgtImgIds.append(ele['image_id'])
fgtImgIds = set(fgtImgIds)

# filtered result analysis
with open('precomputed_coco/filtered_cap_result.json', 'rb') as f:
	fres = json.load(f)
	fannsImgIds = set([ann['image_id'] for ann in fres])
annotation_file = YOUR_LOCAL_PATH + '/annotations/filtered_test_gt.json'
results_file = 'precomputed_coco/filtered_cap_result.json'
coco = COCO(annotation_file)

coco_result = coco.loadRes(results_file)
coco_eval = COCOEvalCap(coco, coco_result)

coco_eval.params['image_id'] = coco_result.getImgIds()

# remove duplicates at image ids = 43635, 51576, 5154, 359
coco_result.imgToAnns[43635] = [coco_result.imgToAnns[43635][0]]
coco_result.imgToAnns[51576] = [coco_result.imgToAnns[51576][0]]
coco_result.imgToAnns[5154] = [coco_result.imgToAnns[5154][0]]
coco_result.imgToAnns[359] = [coco_result.imgToAnns[359][0]]

#########
# Debug #
#########
'''
from tokenizer.ptbtokenizer import PTBTokenizer
gts, res = {}, {}
imgIds = coco_eval.params['image_id']
gts, res = {}, {}
for imgId in imgIds:
	gts[imgId] = coco_eval.coco.imgToAnns[imgId]
	res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]
tokenizer = PTBTokenizer()
gts_id = [k for k, v in gts.items() for _ in range(len(v))]
gts_st = '\n'.join([c['caption'].replace('\n', ' ') for k, v in gts.items() for c in v])
res_id = [k for k, v in res.items() for _ in range(len(v))]
res_st = '\n'.join([c['caption'].replace('\n', ' ') for k, v in res.items() for c in v])
#gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)

#43635, 51576, 5154, 359
imgIds = gts.keys()
for i in imgIds:
	hypo = res[i]
	if len(hypo) != 1:
		print(i)
'''

coco_eval.evaluate()
print('')
print('All scores by pycocoevalcap:')
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score*100:.3f}')
