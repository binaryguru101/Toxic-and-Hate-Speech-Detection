
import csv
import datetime
import json
import os
import random
import time
from argparse import ArgumentParser

import datasets
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from toxcl_soft_tg import (
    ToXCL,
    ToxclDataset,
    build_target_group_vocab,
    encode_target_groups_batch,
)


def initialize_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def load_data(file_path, text_column_num):
    data = []
    with open(file_path, encoding="utf-8") as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            data.append({
                "document": row[text_column_num].strip(),
                "label": row[2].strip(),
                "summary": row[4].strip(),
            })
    return data


def lcs_len(x_tokens, y_tokens):
    m, n = len(x_tokens), len(y_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x_tokens[i] == y_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_score(ref, pred):
    ref_toks = ref.split()
    pred_toks = pred.split()
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        return 0.0
    lcs = lcs_len(ref_toks, pred_toks)
    prec = lcs / max(len(pred_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def meteor_like_score(ref, pred):
    ref_set = ref.split()
    pred_set = pred.split()
    if len(ref_set) == 0 or len(pred_set) == 0:
        return 0.0
    overlap = len(set(ref_set) & set(pred_set))
    prec = overlap / max(len(pred_set), 1)
    rec = overlap / max(len(ref_set), 1)
    if prec + rec == 0:
        return 0.0
    return 10 * prec * rec / (rec + 9 * prec)


def compute_classification_scores(labels, preds):
    acc = accuracy_score(labels, preds) * 100.0
    f1 = f1_score(labels, preds, average="macro") * 100.0
    return round(acc, 2), round(f1, 2)


def compute_generation_scores(labels, preds):
    smoothie = SmoothingFunction().method1
    bleu_scores, rouge_scores, meteor_scores = [], [], []

    for ref, pred in zip(labels, preds):
        ref = ref.strip()
        pred = pred.strip()

        if ref.lower() == "none" and pred.lower() == "none":
            bleu_scores.append(100.0)
            rouge_scores.append(100.0)
            meteor_scores.append(100.0)
            continue

        if ref.lower() == "none" or pred.lower() == "none":
            bleu_scores.append(0.0)
            rouge_scores.append(0.0)
            meteor_scores.append(0.0)
            continue

        ref_toks = ref.split()
        pred_toks = pred.split()
        if len(pred_toks) == 0:
            bleu = 0.0
        else:
            bleu = sentence_bleu([ref_toks], pred_toks, smoothing_function=smoothie) * 100.0
        rouge = rouge_l_score(ref, pred) * 100.0
        meteor = meteor_like_score(ref, pred) * 100.0

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        meteor_scores.append(meteor)

    bertscore = 0.0
    return (
        round(float(np.mean(bleu_scores)), 2),
        round(float(np.mean(rouge_scores)), 2),
        round(float(np.mean(meteor_scores)), 2),
        bertscore,
    )


def main(args):
    initialize_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = f"{args.data_root}/{args.dataset_name}_train.csv"
    valid_file = f"{args.data_root}/{args.dataset_name}_valid.csv"

    decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if decoder_tokenizer.pad_token is None:
        if decoder_tokenizer.eos_token is not None:
            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        else:
            decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

    teacher_tokenizer = None
    teacher_model = None
    if args.teacher_name_or_path:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_name_or_path)
        if teacher_tokenizer.pad_token is None:
            if teacher_tokenizer.eos_token is not None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            else:
                teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_name_or_path,num_labels=2,ignore_mismatched_sizes=True,).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    train_data = load_data(train_file, args.text_column_num)
    valid_data = load_data(valid_file, args.text_column_num)

    tg_vocab = build_target_group_vocab(train_data + valid_data)
    with open(os.path.join(args.output_dir, "tg_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tg_vocab, f, ensure_ascii=False, indent=2)

    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    valid_data = datasets.Dataset.from_pandas(pd.DataFrame(data=valid_data))

    train_dataset = ToxclDataset(train_data)
    valid_dataset = ToxclDataset(valid_data)

    def collate_fn(batch):
        input_texts = [item["document"] for item in batch]
        summary_texts = [item["summary"] for item in batch]
        batch_tgs = [item["target_groups"] for item in batch]

        tokenized_inputs = decoder_tokenizer(
            input_texts,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        labels = decoder_tokenizer(
            summary_texts,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ).input_ids
        labels[labels == decoder_tokenizer.pad_token_id] = -100

        tg_ids, tg_mask = encode_target_groups_batch(batch_tgs, tg_vocab, args.tg_top_k)

        new_batch = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
            "student_cls_labels": torch.as_tensor([item["student_cls_labels"] for item in batch], dtype=torch.long),
            "teacher_cls_labels": torch.as_tensor([item["teacher_cls_labels"] for item in batch], dtype=torch.long),
            "tg_ids": tg_ids,
            "tg_mask": tg_mask,
        }

        if teacher_model is not None:
            teacher_inputs = teacher_tokenizer(
                input_texts,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            new_batch["teacher_input_ids"] = teacher_inputs["input_ids"]
            new_batch["teacher_attention_mask"] = teacher_inputs["attention_mask"]

        return new_batch

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    validation_dataloader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.valid_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print(f"{len(train_dataset):,} training samples")
    print(f"{len(valid_dataset):,} validation samples")

    model = ToXCL(
        decoder_model=decoder_model,
        num_target_groups=len(tg_vocab) + 1,
        pad_token_id=decoder_tokenizer.pad_token_id,
        distill_alpha=args.distill_alpha,
        max_grad_norm=args.max_grad_norm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = max((len(train_dataloader) * args.num_epochs) // args.accumulation_steps, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    training_stats = []
    total_t0 = time.time()
    best_result = float("inf")
    num_step = 0

    if args.resume_training:
        ckpt = model.load_checkpoint(args.output_dir, optimizer=optimizer, scheduler=scheduler)
        training_stats = ckpt.get("training_stats", [])
        if training_stats:
            best_result = training_stats[-1]["Best result"]
            num_step = training_stats[-1]["Step"]

    for epoch_i in range(args.num_epochs):
        print("")
        print(f"======== Epoch {epoch_i + 1} / {args.num_epochs} ========")
        model.train()

        total_train_lm_loss = 0.0
        total_train_cls_loss = 0.0
        total_train_kl_loss = 0.0

        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        for _, batch in train_loop:
            num_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            teacher_logits = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["teacher_input_ids"],
                        attention_mask=batch["teacher_attention_mask"],
                    )
                    teacher_logits = teacher_outputs.logits

            model.zero_grad(set_to_none=True)
            cls_outputs, lm_loss, cls_loss, kl_loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                lm_labels=batch["labels"],
                cls_labels=batch["student_cls_labels"],
                tg_ids=batch["tg_ids"],
                tg_mask=batch["tg_mask"],
                teacher_logits=teacher_logits,
            )

            total_loss = (lm_loss + cls_loss + kl_loss) / args.accumulation_steps
            total_loss.backward()

            total_train_lm_loss += lm_loss.item()
            total_train_cls_loss += cls_loss.item()
            total_train_kl_loss += kl_loss.item()

            if (num_step % args.accumulation_steps == 0) or (num_step == total_steps):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loop.set_postfix(
                loss_lm=round(total_train_lm_loss / max(num_step, 1), 4),
                loss_cls=round(total_train_cls_loss / max(num_step, 1), 4),
                loss_kl=round(total_train_kl_loss / max(num_step, 1), 4),
            )

            if num_step >= args.eval_delay and ((num_step % args.logging_steps == 0) or (num_step == total_steps)):
                print(f"\nRunning validation at step {num_step}...")
                model.eval()

                total_eval_lm_loss = 0.0
                total_eval_cls_loss = 0.0
                total_eval_kl_loss = 0.0
                eval_cls_preds, eval_cls_labels = [], []
                eval_gen_preds, eval_gen_labels = [], []

                for _, vbatch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), leave=False):
                    vbatch = {k: v.to(device) for k, v in vbatch.items()}

                    teacher_logits = None
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                input_ids=vbatch["teacher_input_ids"],
                                attention_mask=vbatch["teacher_attention_mask"],
                            )
                            teacher_logits = teacher_outputs.logits

                    with torch.no_grad():
                        cls_outputs, lm_loss, cls_loss, kl_loss = model(
                            input_ids=vbatch["input_ids"],
                            attention_mask=vbatch["attention_mask"],
                            lm_labels=vbatch["labels"],
                            cls_labels=vbatch["student_cls_labels"],
                            tg_ids=vbatch["tg_ids"],
                            tg_mask=vbatch["tg_mask"],
                            teacher_logits=teacher_logits,
                        )

                        gen_preds = model.generate_expl(
                            input_ids=vbatch["input_ids"],
                            attention_mask=vbatch["attention_mask"],
                            tg_ids=vbatch["tg_ids"],
                            tg_mask=vbatch["tg_mask"],
                            num_beams=4,
                            max_new_tokens=50,
                        )

                    cls_preds = cls_outputs.argmax(dim=-1).detach().cpu().numpy()
                    cls_labels = vbatch["teacher_cls_labels"].detach().cpu().numpy()

                    gen_preds = decoder_tokenizer.batch_decode(gen_preds, skip_special_tokens=True)
                    gen_labels = vbatch["labels"].detach().clone()
                    gen_labels[gen_labels == -100] = decoder_tokenizer.pad_token_id
                    gen_labels = decoder_tokenizer.batch_decode(gen_labels, skip_special_tokens=True)

                    gen_preds = ["none" if cp == 0 else pred for cp, pred in zip(cls_preds, gen_preds)]

                    eval_cls_preds.extend(cls_preds.tolist())
                    eval_cls_labels.extend(cls_labels.tolist())
                    eval_gen_preds.extend([text.split('SEP>')[-1].strip() for text in gen_preds])
                    eval_gen_labels.extend([text.split('SEP>')[-1].strip() for text in gen_labels])

                    total_eval_lm_loss += lm_loss.item()
                    total_eval_cls_loss += cls_loss.item()
                    total_eval_kl_loss += kl_loss.item()

                acc, f1 = compute_classification_scores(eval_cls_labels, eval_cls_preds)
                bleu, rouge, meteor, bertscore = compute_generation_scores(eval_gen_labels, eval_gen_preds)

                avg_valid_lm = total_eval_lm_loss / max(len(validation_dataloader), 1)
                avg_valid_cls = total_eval_cls_loss / max(len(validation_dataloader), 1)
                avg_valid_kl = total_eval_kl_loss / max(len(validation_dataloader), 1)

                print(f"Average valid LM loss:  {avg_valid_lm:.4f}")
                print(f"Average valid CLS loss: {avg_valid_cls:.4f}")
                print(f"Average valid KL loss:  {avg_valid_kl:.4f}")
                print(f"Classification: Acc {acc}, F1 {f1}")
                print(f"Generation: BLEU-4 {bleu}, ROUGE-L {rouge}, METEOR {meteor}, BERTScore {bertscore}")

                training_stats.append({
                    "Step": num_step,
                    "Best result": best_result,
                    "Avg train LM loss": total_train_lm_loss / max(num_step, 1),
                    "Avg train CLS loss": total_train_cls_loss / max(num_step, 1),
                    "Avg train KL loss": total_train_kl_loss / max(num_step, 1),
                    "Avg valid LM loss": avg_valid_lm,
                    "Avg valid CLS loss": avg_valid_cls,
                    "Avg valid KL loss": avg_valid_kl,
                    "CLS evaluation": f"Acc: {acc}, F1: {f1}",
                    "Generation evaluation": f"BLEU-4: {bleu}, ROUGE-L: {rouge}, METEOR: {meteor}, BERTSCORE: {bertscore}",
                })

                if avg_valid_lm < best_result:
                    print(f"New best checkpoint at epoch {epoch_i + 1}, step {num_step}")
                    best_result = avg_valid_lm
                    training_stats[-1]["Best result"] = best_result
                    model.save_checkpoint(
                        os.path.join(args.output_dir, "best_ckpt"),
                        is_best=True,
                    )

                model.save_checkpoint(
                    args.output_dir,
                    is_best=False,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_stats=training_stats,
                )
                with open(os.path.join(args.output_dir, "training_model_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(training_stats, f, indent=2)

                model.train()

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--teacher_name_or_path", type=str, default=None)
    parser.add_argument("--text_column_num", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--eval_delay", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tg_top_k", type=int, default=3)
    parser.add_argument("--distill_alpha", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
