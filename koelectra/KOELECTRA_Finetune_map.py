from pickletools import optimize

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup,logging
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# 0. 디아비스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)

# 1. 경고 메시지 제거
logging.set_verbosity_error()

# 2. 데이터 로드
train_data_path = "map_reviews_train.txt"
dataset = pd.read_csv(train_data_path, sep='\t').dropna(axis=0)
text = list(dataset["document"].values)
label = dataset["label"].values

num_to_print = 3
print("=== 데이터 확인 ===")
for j in range(num_to_print):
    print(f"영화 리뷰:text{[j][:20]}), \t긍부정 라벨 : {label[j]}")
print(f"\t * 학습 데이터의 수: {len(text)}")
print(f"\t * 부정 리뷰 수: {list(label).count(0)}")
print(f"\t * 긍정 리뷰 수: {list(label).count(1)}")

# 3. 텍스트 토큰화
tokenizer = ElectraTokenizer.from_pretrained("koelectra_base_v3-discriminator")
inputs = tokenizer(text, truncation=True, max_length=256, add_special_tokens=True,
                   padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n=== 토큰과 결과 ===")
for j in range(num_to_print):
    print(f"{j+1}번째 데이터")
    print("== 토큰 ==")
    print(input_ids[j])
    print("=== 어텐션 마스크 ===")
    print(attention_mask[j])

# 4. 데이터 분리
train, valid, train_y, valid_y = train_test_split(input_ids, label, test_size=0.2,
                                                  random_state=2026)
train_masks, valid_masks, _, _ = train_test_split(attention_mask, label, test_size=0.2,
                                                 random_state=2026)
# 5. python의 데이터로더 설명
batch_size = 32
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_mask = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_labels, train_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_inputs = torch.tensor(valid)
valid_labels = torch.tensor(valid_y)
valid_masks = torch.tensor(valid_masks)
valid_data = TensorDataset(valid_inputs, valid_labels, valid_masks)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

# 6. 모델 성정
model = ElectraForSequenceClassification.from_pretrained('koelectra_base_v3-discriminator',
                                                         num_labels=4)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, eps=1e-06, betas=(0.9, 0.999))

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)

epoch_results = []

# 7. 학습 진행 (중간에 학습 데이터와 검증 데이터의 정확도를 계산)
for e in range(0, epoch):
    # === 학습시작
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Traing Epoch {e+1}", leave=False)

    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        model.zero_grad()

        outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_label)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    # ==== 학습 종료

    # ===== 학습 데이터에 대한 정확도 측정
    model.eval()
    progress_bar = tqdm(train_dataloader, desc=f"Evaluating Train Epoch {e+1}", leave=False)
    train_preds = []; train_true = []
    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch_label.cpu().numpy())
    train_accuracy = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)
    # ===== 학습 데이터에 대한 정확도 측정 완료
    progress_bar = tqdm(valid_dataloader, desc=f"Evaluating Train Epoch {e + 1}", leave=False)
    valid_preds = []; valid_true = []
    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        valid_preds.extend(preds.cpu().numpy())
        valid_true.extend(batch_label.cpu().numpy())
    valid_accuracy = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)
    # ===== 검증 데이터에 대한 정확도 측정 완료
    epoch_results.append((avg_train_loss, train_accuracy, valid_accuracy))

print("\n === 학습 결과 요약 ===")
for j, (loss, t_acc, v_acc) in enumerate(epoch_results, start=1):
    print(f"Epoch: {j}, Train loss: {loss:.4f}, Train ACC:{t_acc:.4f}, Valid ACC:{v_acc:.4f}")

print("\n === 모델 저장 ===")
save_path = "koelectra-base-map"
model.cpu()
for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()
model.save_pretrained(save_path,'.pt')
print("\n=== 종료 ===")
