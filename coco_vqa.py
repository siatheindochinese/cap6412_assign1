import torch
from PIL import Image
import numpy as np
import os
import json

from lavis.models import load_model_and_preprocess
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from datasets import load_dataset

device='cuda'

################
# Load Dataset #
################
YOUR_LOCAL_PATH = 'cache/coco'
id_path_map = load_dataset(YOUR_LOCAL_PATH, data_files={'val':'annotations/vqa_val_eval.json'})['val']
coco_val = load_dataset(YOUR_LOCAL_PATH, data_files={'val':'annotations/v2_OpenEnded_mscoco_val2014_questions.json'})['val']

#############################
# Load Model and Processors #
#############################
if 'vqa_result.json' not in os.listdir('precomputed_coco'):
	model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
	
################################
# Preprocess data for vqa task #
################################
question_id = []
qidtoimgpth = {}
question = []

# answer in this format: {'answer': '3', 'question_id': 4870250}

# extract question id and question
for ann in coco_val['questions'][0]:
	ques = ann['question']
	quesid = ann['question_id']
	question.append(ques)
	question_id.append(quesid)
	
# extract question to imagepth mapping
for ele in id_path_map:
	quesid = ele['question_id']
	imgpth = ele['image']
	qidtoimgpth[quesid]=imgpth
	
###################################################################
# Generate an answer for all visual questions, then store in json #
###################################################################
# in total there are 200,000+ questions
# only k questions will be evaluated
k = 50000
results = []
if 'vqa_result.json' not in os.listdir('precomputed_coco'):
	for i in range(k):
		print('i:',i)
		quesid = question_id[i]
		ques = question[i]
		imgpth = qidtoimgpth[quesid]
		
		ques = f"Question: {ques} Short answer:"
		
		raw_image = Image.open(YOUR_LOCAL_PATH+'/images/'+imgpth).convert("RGB")
		image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		question_processed = txt_processors["eval"](ques)
		ans = model.predict_answers(samples={"image": image_processed, "text_input": question_processed}, inference_method="generate")
		print(ans)
		results.append({'answer': ans[0], 'question_id': quesid})
	with open('precomputed_coco/vqa_result.json', 'w') as f:
		json.dump(results, f)
		
###################################
# Load and preprocess annotations #
###################################
resFile = 'precomputed_coco/vqa_result.json'
quesFile = YOUR_LOCAL_PATH + '/annotations/v2_OpenEnded_mscoco_val2014_questions.json'
annFile = YOUR_LOCAL_PATH + '/annotations/v2_mscoco_val2014_annotations.json'

res = json.load(open(resFile))
ques = json.load(open(quesFile))
anns = json.load(open(annFile))

#for debugging
resQuesIds = set([ann['question_id'] for ann in res])
quesQuesIds = set([ann['question_id'] for ann in ques['questions']])
annsQuesIds = set([ann['question_id'] for ann in anns['annotations']])

overlap = resQuesIds & quesQuesIds & annsQuesIds
i = 0
if 'filtered_v2_OpenEnded_mscoco_val2014_questions.json' not in os.listdir(YOUR_LOCAL_PATH + '/annotations'):
	for quesid in quesQuesIds:
		print(i)
		i+=1
		if quesid not in overlap:
			print(quesid,'not in quesQuesIds')
			for ann in ques['questions']:
				if ann['question_id'] == quesid:
					ques['questions'].remove(ann)
		if quesid not in overlap:
			print(quesid,'not in annsQuesIds')
			for ann in anns['annotations']:
				if ann['question_id'] == quesid:
					anns['annotations'].remove(ann)
		print('')
	
	with open(YOUR_LOCAL_PATH + '/annotations/filtered_v2_OpenEnded_mscoco_val2014_questions.json','w') as f:
		json.dump(ques, f)
	with open(YOUR_LOCAL_PATH + '/annotations/filtered_v2_mscoco_val2014_annotations.json','w') as f:
		json.dump(anns, f)

################
# VQA Pipeline #
################
resFile = 'precomputed_coco/vqa_result.json'
quesFile = YOUR_LOCAL_PATH + '/annotations/filtered_v2_OpenEnded_mscoco_val2014_questions.json'
annFile = YOUR_LOCAL_PATH + '/annotations/filtered_v2_mscoco_val2014_annotations.json'

vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

vqaEval = VQAEval(vqa, vqaRes, n=2)
vqaEval.evaluate()

print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
