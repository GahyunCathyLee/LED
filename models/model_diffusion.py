import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

from models.layers import PositionalEncoding, ConcatSquashLinear


class social_transformer(nn.Module):
	def __init__(self, t_h=10, d_h=6):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(t_h * d_h, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: (B*N, T_H, d_h)
		mask: (B, N, N)
		'''
		# 1. 특징 추출 (B*N, 256)
		h_feat = self.encode_past(h.reshape(h.size(0), -1))
		
		# 2. [수정] B(배치)와 N(노드 수)을 복원하여 시퀀스 형태로 변환
		# batch_first=True이므로 (B, N, 256)이 됩니다.
		B = mask.size(0)
		h_seq = h_feat.view(B, -1, 256) 
		
		# 3. Transformer 적용 (B, N, N) 마스크와 (B, N, 256) 데이터 연산
		h_feat_seq = self.transformer_encoder(h_seq, mask)
		
		# 4. 결과 리턴: 잔차 연결 후 다시 (B*N, 1, 256)으로 펼침
		return (h_seq + h_feat_seq).reshape(-1, 256).unsqueeze(1)


class TransformerDenoisingModel(Module):

	def __init__(self, context_dim=256, tf_layer=2, t_h=10, d_h=6):
		super().__init__()
		self.encoder_context = social_transformer(t_h=t_h, d_h=d_h)
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=100)
		self.concat1 = ConcatSquashLinear(2, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)


	def forward(self, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		x = self.pos_emb(x)

		trans = self.transformer_encoder(x)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans)
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask):
		# x: (B*N, n_samples, T_F, 2)   where n_samples = K // 2
		n_samples = x.size(1)   # K // 2  (= 10 when k_pred=20)
		t_f       = x.size(2)   # T_F  (dynamic: 20 for NBA, 25 for highD, ...)

		beta = beta.view(beta.size(0), 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)   # (B*N, 1, 256)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B*N, 1, 3)
		ctx_emb  = torch.cat([time_emb, context], dim=-1).repeat(1, n_samples, 1).unsqueeze(2)
		# ctx_emb: (B*N, n_samples, 1, 259)

		x = self.concat1.batch_generate(ctx_emb, x)
		x_flat = x.view(-1, t_f, 512)
		final_emb = self.pos_emb(x_flat)

		trans_flat = self.transformer_encoder(final_emb)
		trans = trans_flat.view(-1, n_samples, t_f, 512)
		# (B*N, n_samples, T_F, 512)
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		return self.linear.batch_generate(ctx_emb, trans)
