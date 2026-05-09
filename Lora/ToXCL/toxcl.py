import json
import os
from ast import literal_eval

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

id2label = {0: "normal", 1: "hate"}
label2id = {"normal": 0, "hate": 1}

class ToXCL(nn.Module):
    def __init__(self, decoder_model, decoder_tokenizer=None, tg_model=None, tg_tokenizer=None, hidden_size=768, num_labels=2):
        super(ToXCL, self).__init__()
        self.device = decoder_model.device

        # --- LoRA wrapping (NOVELTY) ---
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q","v"]
            )
        self.decoder_model = get_peft_model(decoder_model, lora_config)
        self.decoder_model.print_trainable_parameters()
        # --------------------------------

        self.decoder_tokenizer = decoder_tokenizer
        self.tg_model = tg_model
        self.tg_tokenizer = tg_tokenizer
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.device)
        self.loss_fct = nn.BCELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.activation = nn.Softmax(dim=-1)

    def classify(self, input_ids, attention_mask=None):
        outputs = self.decoder_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        last_hidden_state = outputs.encoder_last_hidden_state
        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()
        logits = self.classifier(cls_token_emb).squeeze()
        logits = logits.view(-1, self.num_labels)
        return self.activation(logits)

    def generate_tg(self, **kwargs):
        return self.tg_model.generate(**kwargs)

    def generate_expl(self, **kwargs):
        return self.decoder_model.generate(**kwargs)

    def generate_e2e(self, prompts, apply_constraints=True, tg_generation_params=None, explanation_params=None, **kwargs):
        tg_prompts = ["summarize: " + p for p in prompts]
        tg_inputs = self.tg_tokenizer(tg_prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)

        tg_outputs = self.tg_model.generate(
            input_ids=tg_inputs["input_ids"],
            attention_mask=tg_inputs["attention_mask"],
            **tg_generation_params
        )
        decoded_tg = self.tg_tokenizer.batch_decode(tg_outputs, skip_special_tokens=True)

        student_prompts = [f"Target: {tg} Post: {p}" for tg, p in zip(decoded_tg, prompts)]
        inputs = self.decoder_tokenizer(student_prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)

        logits = self.classify(inputs["input_ids"], inputs["attention_mask"])
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        prediction_labels = [id2label[p] for p in pred_ids]

        expl_outputs = self.decoder_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **explanation_params
        )
        decoded_expl = self.decoder_tokenizer.batch_decode(expl_outputs, skip_special_tokens=True)

        final_explanations = []
        for label, expl, original_text in zip(prediction_labels, decoded_expl, prompts):
            if label == "normal" and apply_constraints:
                final_explanations.append("none")
            else:
                clean_expl = expl.replace("hate SEP>", "").replace("SEP>", "").strip()

                if not clean_expl:
                    final_explanations.append("none")
                else:
                    final_explanations.append(clean_expl)

        return {
            "target_groups": decoded_tg,
            "detections": prediction_labels,
            "explanations": final_explanations
        }


    def forward(self, input_ids=None, attention_mask=None, lm_labels=None, cls_labels=None, teacher_logits=None, alpha=0.2):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        lm_labels = lm_labels.to(self.device)
        cls_labels = cls_labels.to(self.device)
        lm_outputs = self.decoder_model(input_ids, attention_mask=attention_mask, labels=lm_labels)
        lm_loss = lm_outputs.loss
        last_hidden_state = lm_outputs.encoder_last_hidden_state
        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()
        cls_logits = self.classifier(cls_token_emb).squeeze().to(self.device)
        cls_logits = cls_logits.view(-1, self.num_labels)
        cls_outputs = self.activation(cls_logits)

        if teacher_logits is not None:
            teacher_output = self.activation(teacher_logits)
            cls_loss = self.loss_fct(cls_outputs, alpha*teacher_output + (1-alpha)*cls_labels.float())
            kl_loss = self.kl_loss(cls_outputs, teacher_output)
        else:
            cls_loss = self.loss_fct(cls_outputs, cls_labels.float())
            kl_loss = torch.tensor(0.0, device=self.device)

        return cls_outputs, lm_loss, cls_loss, kl_loss

    def save_checkpoint(self, output_dir, is_best=False, optimizer=None, scheduler=None, training_stats=None):
        os.makedirs(output_dir, exist_ok=True)
        self.decoder_model.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
        if not is_best:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            with open(os.path.join(output_dir, "training_model_stats.json"), "w") as file:
                json.dump(training_stats, file)

    def load_checkpoint(self, output_dir, base_model_path="google/flan-t5-base", optimizer=None, scheduler=None):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
        self.decoder_model = PeftModel.from_pretrained(base_model, output_dir).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(output_dir, "classifier.pt")))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
        print("Successfully loaded checkpoint.")


class ToxclDataset(Dataset):
    def __init__(self, data):
        self.inputs = ["summarize: " + doc for doc in data["document"]]
        self.outputs = data["summary"]
        encoded_labels = [label2id[i] for i in data["label"]]
        self.student_cls_labels = [[1,0] if int(i)==0 else [0,1] for i in encoded_labels]
        self.teacher_cls_labels = [int(i) for i in encoded_labels]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        output = self.outputs[idx]
        try:
            output = literal_eval(output)
        except:
            output = [output]
        label = np.random.choice(output)
        return dict(
            document=self.inputs[idx],
            label=label,
            student_cls_labels=self.student_cls_labels[idx],
            teacher_cls_labels=self.teacher_cls_labels[idx]
        )
