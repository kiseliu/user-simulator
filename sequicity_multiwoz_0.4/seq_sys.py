#!/bin/usr/env python3
#
import sys, os, re, pdb
# sys.path.append('/home/wyshi/simulator/sequcity_user/')
# sys.path.append('/data/qkun/sequcity_mulitwoz_0.4/')
sys.path.append('/home/wyshi/simulator/')
import logging, random
import torch
import numpy as np
from nltk import word_tokenize
from collections import defaultdict

from sequicity_user.model import Model
from sequicity_user.config import global_config as cfg

import simulator.dialog_config as dialog_config
import simulator.nlg as nlg
from simulator.system import System
from simulator.agent.core import Action, SystemAct

class Seq_System(System):

	def __init__(self):
		super().__init__()

		# # # # # # # # # # # # # # # # 
		# # model configure setting # #
		cfg.init_handler('tsdf-sys')
		cfg.dataset = 'sys'
		if cfg.cuda:
			torch.cuda.set_device(cfg.cuda_device)
			logging.info('Device: {}'.format(torch.cuda.current_device()))
		self.m = Model('sys')
		self.m.count_params()
		self.m.load_model()
		self.entity = self.m.reader.entity
		# # # # # # # # # # # # # # # # 

		self._set_initial_model_parameters()

	def _set_initial_model_parameters(self):
		self.turn_batch = {
				'dial_id': [0],
				'turn_num': [0],
				'user': [[0]],
				'response': [[0]],
				'bspan': [[0]],
				'u_len': [0],
				'm_len': [0],
				'degree': [[1]],
				'supervised': [True],
				'goal': [self.m.reader.vocab.sentence_encode(word_tokenize(' '.join(self.goal_list)) + ['EOS_Z0'])]
		}
		self.prev_z = None
		self.prev_m = None

	def respond(self):
		
		mode = 'test'
		turn_states = {}
		turn_num = self.turn_batch['turn_num'][0]

		if prev_sys is None:
			prev_sys = 'Hello! What can I help you?'.lower()
		else:
			prev_sys = prev_sys.lower()

		# # format input
		utt_tokenized = word_tokenize(prev_sys) + ['EOS_U']
		utt_encoded   = self.m.reader.vocab.sentence_encode(utt_tokenized)

		if self.turn_batch['turn_num'] == [0]:
			self.turn_batch['user'] = [utt_encoded]
		else:
			self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_m)) + \
								 [self.m.reader.vocab.encode('EOS_M')] + \
								 utt_encoded]

		self.turn_batch['u_len'] = [len(i) for i in self.turn_batch['user']]
		self.turn_batch['m_len'] = [len(i) for i in self.turn_batch['response']]

		u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
			m_len, degree_input, kw_ret \
				= self.m._convert_batch(self.turn_batch, self.prev_z)

		# # execute tsd-net
		m_idx, z_idx, turn_states = self.m.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
										   m_input=m_input,
										   degree_input=degree_input, u_input_np=u_input_np,
										   m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
										   dial_id=self.turn_batch['dial_id'], **kw_ret)

		cur_usr = self.m.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')
		filled_usr = self.fill_sentence(cur_usr)

		turn_num += 1
		self.prev_z = z_idx
		self.prev_m = filled_usr
		turn_num += 1
		self.turn_batch['turn_num'] = [turn_num]
		# self.turn_batch['bspan'] = self.prev_z

		return None, self.prev_m

	def fill_sentence(self, slot_sent):
		sent = []
		for word in word_tokenize(slot_sent):
			if '_SLOT' in word:
				slot_name = word.split('_')[0]
				if slot_name not in self.goal_dic:
					slot_val = random.choice(self.entity['informable'][slot_name])
					# pdb.set_trace()
				else:

					if len(self.goal_dic[slot_name]) > 1:
						slot_val = self.goal_dic[slot_name].pop(0)
					else:
						slot_val = self.goal_dic[slot_name][0]
				sent.append(slot_val)
			else:
				sent.append(word)				
		return ' '.join(sent)

	def reset(self):
		super().reset()

		self._set_initial_model_parameters()


def main():
	sys = Seq_System():
	sys.interact()


if __name__ == "__main__":
	main()
