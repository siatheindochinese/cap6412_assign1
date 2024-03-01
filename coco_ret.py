import torch
from PIL import Image
import numpy as np
import os

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from datasets import load_dataset

device='cuda'

################
# Load Dataset #
################
YOUR_LOCAL_PATH = 'cache/coco'
coco_test = load_dataset(YOUR_LOCAL_PATH, data_files={'test':'annotations/coco_karpathy_test.json'})['test']

#############################
# Load Model and Processors #
#############################
'''
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
'''

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)

######################################
# Preprocess data for retrieval task #
######################################
text = []
image = []
txt2img = {}
img2txt = {}

txt_id = 0
for img_id, ann in enumerate(coco_test):
	# evaluate on a subset of COCO to save computation
	if img_id == 1000:
		break
	image.append(ann["image"])
	img2txt[img_id] = []
	for i, caption in enumerate(ann["caption"]):
		text.append(caption)
		img2txt[img_id].append(txt_id)
		txt2img[txt_id] = img_id
		txt_id += 1

##################################################################
# Generate ITC score matrix and get top k=128 t2i and i2t scores #
##################################################################
'''
We follow Section 4.4 of the Blip2 paper which is inspired by ALBEF.
ITM is done on the top k=128 ITC matches for t2i and i2t.
According to the paper, this is done to save memory and computation.
We k=64 due to memory and time constraints.
'''
# get n(i)-by-n(t) matrix to store ITC scores
# compute all combinations
k = 64
itc = np.zeros((len(image),len(text)))

if 'itc.txt' not in os.listdir('precomputed_coco'):
	images_embedding = None
	texts_embedding = None
	multimodal_embedding = None

	# extract image embeddings
	for i in range(len(image)):
		print('i:',i)
		raw_image = Image.open(YOUR_LOCAL_PATH+'/images/'+image[i]).convert("RGB")
		image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		sample = {"image": image_processed, "text_input": None}
		image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
		if images_embedding is None:
			images_embedding = image_emb
		else:
			images_embedding = torch.cat((images_embedding, image_emb),0)

	# extract text embeddings
	for i in range(len(text)):
		print('t:',i)
		caption = text[i]
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_coco/itc.txt', itc)
else:
	itc = np.loadtxt('precomputed_coco/itc.txt')
	
# filter top k=128 for i2t
# i2t, matrix of n(i)x128
# elements are the top k=128 text indices
if 'i2t_idx.txt' not in os.listdir('precomputed_coco'):
	i2t_idx = np.argpartition(itc, -k)[:,-k:]
	np.savetxt('precomputed_coco/i2t_idx.txt', i2t_idx)
else:
	i2t_idx = np.loadtxt('precomputed_coco/i2t_idx.txt').astype(int)
	
# filter top k=128 for t2i 
# i2t, matrix of n(t)x128
# elements are the top k=128 image indices
if 't2i_idx.txt' not in os.listdir('precomputed_coco'):
	t2i_idx = np.argpartition(itc.T, -k)[:,-k:]
	np.savetxt('precomputed_coco/t2i_idx.txt', t2i_idx)
else:
	t2i_idx = np.loadtxt('precomputed_coco/t2i_idx.txt').astype(int)

############################################################
# Generate ITM score matrix and get R1, R5 and R10 metrics #
############################################################
'''
We follow Section 4.4 of the Blip2 paper which is inspired by ALBEF.
ITM is done on the top k=128 ITC matches for t2i and i2t.
According to the paper, this is done to save memory and computation.
'''
# Load ITM model
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)

# ITM score for i2t
itm_i2t = np.zeros((len(image), k))
if 'itm_i2t.txt' not in os.listdir('precomputed_coco'):
	for i in range(len(image)):
		for j in range(k):
			print(i,j)
			t  = i2t_idx[i][j]
			raw_image = Image.open(YOUR_LOCAL_PATH+'/images/'+image[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			caption = text[t]
			text_processed = txt_processors["eval"](caption)
			itm_output = model({"image": image_processed, "text_input": text_processed}, match_head="itm")
			itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
			itm_i2t[i][j] = itm_scores[:, 1].item()
	np.savetxt('precomputed_coco/itm_i2t.txt', itm_i2t)
else:
	itm_i2t = np.loadtxt('precomputed_coco/itm_i2t.txt')

# ITM score for t2i
itm_t2i = np.zeros((len(text), k))
if 'itm_t2i.txt' not in os.listdir('precomputed_coco'):
	for i in range(len(text)):
		for j in range(k):
			print(i,j)
			t  = t2i_idx[i][j]
			raw_image = Image.open(YOUR_LOCAL_PATH+'/images/'+image[t]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			caption = text[i]
			text_processed = txt_processors["eval"](caption)
			itm_output = model({"image": image_processed, "text_input": text_processed}, match_head="itm")
			itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
			itm_t2i[i][j] = itm_scores[:, 1].item()
	np.savetxt('precomputed_coco/itm_t2i.txt', itm_t2i)
else:
	itm_t2i = np.loadtxt('precomputed_coco/itm_t2i.txt')

#compute TR1, 5 and 10 metrics
ranks = np.zeros(itm_i2t.shape[0])
for index, score in enumerate(itm_i2t):
	inds = np.argsort(score)[::-1]
	inds = i2t_idx[index][inds]
	# Score
	rank = 1e20
	for i in img2txt[index]:
		if i not in inds:
			continue
		tmp = np.where(inds == i)[0][0]
		if tmp < rank:
			rank = tmp
	ranks[index] = rank

tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

#compute IR1, 5 and 10 metrics
ranks = np.zeros(itm_t2i.shape[0])
for index, score in enumerate(itm_t2i):
	inds = np.argsort(score)[::-1]
	inds = t2i_idx[index][inds]
	# Score
	rank = 1e20
	i = txt2img[index]
	if i in inds:
		tmp = np.where(inds == i)[0][0]
		if tmp < rank:
			rank = tmp
	ranks[index] = rank

ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

print('TR@1:', round(tr1,1), 'TR@5:', round(tr5,1), 'TR@10:', round(tr10,1))
print('IR@1:', round(ir1,1), 'IR@5:', round(ir5,1), 'IR@10:', round(ir10,1))
