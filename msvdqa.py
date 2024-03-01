import torch
import random
from PIL import Image
import numpy as np
import os
import json
import cv2

from lavis.models import load_model_and_preprocess
from lavis.common.vqa_tools.vqa_eval import VQAEval
from datasets import load_dataset

device = 'cuda'

YOUR_LOCAL_PATH = 'cache/msvdqa'
msvdqa_val = load_dataset(YOUR_LOCAL_PATH, data_files={'val':'annotations/qa_val.json'})['val']

#########################################
# Video loader class for loading videos #
#########################################
class VidLoader:
	def __init__(self, vidpth):
		self.vidpth = vidpth
		
		cap = cv2.VideoCapture(vidpth)
		ret = True
		self.frames = []
		
		while ret:
			ret, frame = cap.read()
			if not ret:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			self.frames.append(frame)
		
		self.numf = len(self.frames)
			
	def sample_rand(self):
		# randomly sample a frame from self.frames
		r = random.randrange(self.numf)
		return self.frames[r]
		
	def sample_uni(self, n=10):
		# uniformly sample n frames from self.frames:
		out = []
		for i in range(0, self.numf, n):
			out.append(self.frames[i])
		return out
		
################################################################
# Wrapper for model to pool image features and generate answer #
################################################################
def temporal_pool_predict_answers(frames, text_input, model, device):
	num_beams=5
	max_len=10
	min_len=1
	length_penalty=0
	with model.maybe_autocast():
		image_embeds = None
		for frame in frames:
			if image_embeds == None:
				image_embeds = model.ln_vision(model.visual_encoder(frame))
			else:
				image_embeds += model.ln_vision(model.visual_encoder(frame))
		image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
			device
		)

		query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
		query_output = model.Qformer.bert(
			query_embeds=query_tokens,
			encoder_hidden_states=image_embeds,
			encoder_attention_mask=image_atts,
			return_dict=True,
		)

		inputs_opt = model.opt_proj(query_output.last_hidden_state)
		atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
			device
		)

		if isinstance(text_input, str):
			text_input = [text_input]
		else:
			text_input = text_input

		model.opt_tokenizer.padding_side = "left"
		opt_tokens = model.opt_tokenizer(
			text_input,
			return_tensors="pt",
			padding="longest",
			truncation=True,
			max_length=model.max_txt_len,
		).to(device)

		attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

		# require transformers>=4.27
		inputs_embeds = model.opt_model.get_input_embeddings()(opt_tokens.input_ids)
		inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)

		outputs = model.opt_model.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			do_sample=False,
			num_beams=num_beams,
			max_new_tokens=max_len,
			min_length=min_len,
			eos_token_id=model.eos_token_id,
			length_penalty=length_penalty,
		)
		output_text = model.opt_tokenizer.batch_decode(
			outputs, skip_special_tokens=True
		)
		output_text = [text.strip() for text in output_text]

	return output_text

#############################
# Load Model and Processors #
#############################
if 'vdqa_result_10frame.json' not in os.listdir('precomputed_msvd'):
	model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)

#################################
# Preprocess data for vdqa task #
#################################
videopth = os.listdir(YOUR_LOCAL_PATH + '/video')
videopth.sort()
videopth = list(map(lambda x:YOUR_LOCAL_PATH + '/video/' + x, videopth))

qidlst = [ele['question_id'] for ele in msvdqa_val]
queslst = [ele['question'] for ele in msvdqa_val]
anslst = [ele['answer'] for ele in msvdqa_val]
qidtovideopth = {}
for ele in msvdqa_val:
	if ele['question_id'] not in qidtovideopth:
		qidtovideopth[ele['question_id']] = YOUR_LOCAL_PATH + '/video/' + ele['video']

###################################################################
# Randomly sample a frame from each video                         #
# Generate an answer for all visual questions, then store in json #
###################################################################
results = []
if 'vdqa_result_1frame.json' not in os.listdir('precomputed_msvd'):
	for i in range(len(qidlst)):
		print('i:',i)
		
		qid = qidlst[i]
		ques = queslst[i]
		ansgt = anslst[i]
		
		vidpth = qidtovideopth[qid]
		vidloader = VidLoader(vidpth)
		
		ques = f"Question: {ques} Short answer:"
		print(ques)
		
		raw_image = Image.fromarray(vidloader.sample_rand())
		image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		question_processed = txt_processors["eval"](ques)
		ans = model.predict_answers(samples={"image": image_processed, "text_input": question_processed}, inference_method="generate")
		print(ans)
		results.append({'prediction': ans[0], 'target': ansgt})
		
	with open('precomputed_msvd/vdqa_result_1frame.json', 'w') as f:
		json.dump(results, f)

with open('precomputed_msvd/vdqa_result_1frame.json', 'rb') as f:
	results = json.load(f)

vqa_tool = VQAEval()

predictions = np.array([vqa_tool.processDigitArticle(vqa_tool.processPunctuation(res["prediction"])) for res in results])
targets = np.array([vqa_tool.processDigitArticle(vqa_tool.processPunctuation(res["target"])) for res in results])
accuracy = (targets == predictions).sum() / targets.shape[0]
print('1 frame accuracy:', round(accuracy * 100, 2),'%')
###################################################################
# temporally pool a uniform sample of frames from each video      #
# Generate an answer for all visual questions, then store in json #
###################################################################
# in total there are 13157 questions
results = []
i = 0
if 'vdqa_result_10frame.json' not in os.listdir('precomputed_msvd'):
	for i in range(len(qidlst)):
		print('i:',i)
		
		qid = qidlst[i]
		ques = queslst[i]
		ansgt = anslst[i]
		
		vidpth = qidtovideopth[qid]
		vidloader = VidLoader(vidpth)
		
		ques = f"Question: {ques} Short answer:"
		print(ques)
		
		frames = vidloader.sample_uni()
		frames = [Image.fromarray(frame) for frame in frames]
		frames = [vis_processors["eval"](frame).unsqueeze(0).to(device) for frame in frames]
		question_processed = txt_processors["eval"](ques)
		ans = temporal_pool_predict_answers(frames, question_processed, model, device)
		print(ans)
		results.append({'prediction': ans[0], 'target': ansgt})
	with open('precomputed_msvd/vdqa_result_10frame.json', 'w') as f:
		json.dump(results, f)
		
with open('precomputed_msvd/vdqa_result_10frame.json', 'rb') as f:
	results = json.load(f)

predictions = np.array([vqa_tool.processDigitArticle(vqa_tool.processPunctuation(res["prediction"])) for res in results])
targets = np.array([vqa_tool.processDigitArticle(vqa_tool.processPunctuation(res["target"])) for res in results])
accuracy = (targets == predictions).sum() / targets.shape[0]
print('10 frame accuracy:', round(accuracy * 100, 2),'%')
