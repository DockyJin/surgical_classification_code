import os
import sys
import socket
import re
import pandas as pd
import torch
import argparse

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,jaccard_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from preprocess_data import load_data_from_folder, preprocess_df
from design_prompt import build_io_prompt, build_cot_prompt, build_tot_prompt

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Surgical Phase Classification using LLMs.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model (default: cuda)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/home/hpc/iwb9/iwb9102h/surgical_phase_code/datadoctor",
        help="Path to the data folder containing input CSVs"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use for evaluation (default: use all)"
    )
    parser.add_argument(
        "--oov_as_other",
        action="store_true",
        help="If enabled, unmatched stages are treated as other tags; otherwise, they are skipped during evaluation."
    )
    parser.add_argument(
        "--strict_match",
        action="store_true",
        help="If enabled, regular expressions are used for stricter matching to avoid mismatching of local substrings."
    )
    return parser.parse_args()


def print_system_info():
    """
    Print system information for debugging.
    """
    print("HOSTNAME:", socket.gethostname())
    print("Which python:", sys.executable)
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
       print("Current device:", torch.cuda.current_device())
       print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 0. Configuration section
# define phase_label
ALL_PHASES = [
    "0-preparation",
    "1-puncture",
    "2-GuideWire",
    "3-CathPlacement",
    "4-CathPositioning",
    "5-CathAdjustment",
    "6-CathControl",
    "7-Closing",
    "8-Transition"
]

PHASE2IDX = {phase: i for i, phase in enumerate(ALL_PHASES)}
IDX2PHASE = {i: phase for phase, i in PHASE2IDX.items()}



# ========== Initialize Hugging Face Model ==========

def load_hf_model(model_dir,device="cuda"):
    """
    Load the local Hugging Face model and tokenizer with 8-bit quantization.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            load_in_8bit=True,
        )
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Model loading failed:{str(e)}")
        sys.exit(1)
    


# ========== Calling the local model for text generation ==========

def call_llm_with_prompt_hf(prompt, tokenizer, model, max_new_tokens=128, device="cuda"):
    """
    Use the local Hugging Face model for inference and return the generated text.
    """
    # Coding Prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
        )
    # decoding
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# ========== Map model return text to fixed phase ==========

def map_output_to_phase(output_text, all_phases=ALL_PHASES,strict_match=False):
    """
    Perform simple string matching on the text returned by the model and map it to the given stage label.
    """
    output_text_lower = output_text.lower()
    
    if strict_match:
        for phase in all_phases:
            pattern = r"\b" + re.escape(phase.lower()) + r"\b"
            if re.search(pattern, output_text_lower):
                return phase
    else:
    # First directly match the stage name
        for phase in all_phases:
            if phase.lower() in output_text_lower:
                return phase
    
    # Try matching the numbers again
    for i, phase in enumerate(all_phases, start=0):
        if str(i) in output_text_lower:
            return phase
    
    return "No matching phase!"


# ========== Main process: Use three kinds of prompts to evaluate the test set text ==========

def main_evaluation(folder_path, model_dir, sample_size=None, device="cuda",oov_as_other=False,strict_match=False):
    """
    Main function: load data -> preprocess -> 
    generate three prompts -> call local Hugging Face model -> evaluate results
     """
    # 1) Read and merge all CSVs
    df = load_data_from_folder(folder_path)
    if df.empty:
        raise ValueError("Empty dataframe after loading")
    
    # 2) Preprocessing
    df = preprocess_df(df)
    
    # 3) If the amount of data is large, you can first sample
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Make sure there are two columns in the data: Text and Phase_Label
    if "Text" not in df.columns or "Phase_Label" not in df.columns:
        print("CSV data does not contain Text or Phase_Label, cannot be evaluated!")
        return

    # Loading Model
    print(f"Loading model from directory:{model_dir} ...")
    tokenizer, model = load_hf_model(model_dir, device="cuda")

    results_io = []
    results_cot = []
    results_tot = []
    gold_labels = []

    # 4) Traverse each data and adjust 3 prompts respectively
    for idx, row in df.iterrows():
        text = row["Text"]
        gold_label = row["Phase_Label"]  # Assume that it corresponds to one of ALL_PHASES  
        gold_labels.append(gold_label)

        # -- IO Prompt --
        prompt_io = build_io_prompt(text)
        output_io = call_llm_with_prompt_hf(prompt_io, tokenizer, model, device=device)
        pred_phase_io = map_output_to_phase(output_io, ALL_PHASES)
        results_io.append(pred_phase_io)

        # -- CoT Prompt --
        prompt_cot = build_cot_prompt(text)
        output_cot = call_llm_with_prompt_hf(prompt_cot, tokenizer, model, device=device)
        pred_phase_cot = map_output_to_phase(output_cot, ALL_PHASES)
        results_cot.append(pred_phase_cot)

        # -- ToT Prompt --
        prompt_tot = build_tot_prompt(text)
        output_tot = call_llm_with_prompt_hf(prompt_tot, tokenizer, model, device=device)
        pred_phase_tot = map_output_to_phase(output_tot, ALL_PHASES)
        results_tot.append(pred_phase_tot)
    
    # ========== 5) Indicators such as recognition accuracy in the evaluation phase ==========
    extended_phase_list = ALL_PHASES.copy()
    if oov_as_other and "other" not in extended_phase_list:
        extended_phase_list.append("other")
    extended_phase2idx = {phase: i for i, phase in enumerate(extended_phase_list)}
    
    def map_to_idx(label_str):
        """
        Adapts for the 'No matching phase!' case, returns the index of 'other' if oov_as_other is True;
otherwise returns -1.
        """
        if label_str == "No matching phase!":
            if oov_as_other:
                return extended_phase2idx["other"]
            else:
                return -1
        return extended_phase2idx.get(label_str, -1)

    gold_indices = [map_to_idx(lbl) for lbl in gold_labels]
    io_indices   = [map_to_idx(lbl) for lbl in results_io]
    cot_indices  = [map_to_idx(lbl) for lbl in results_cot]
    tot_indices  = [map_to_idx(lbl) for lbl in results_tot]
    
    # If OOV is not considered as 'other', you need to filter out the samples with -1 before calculating the index
    if not oov_as_other:
    # ---------- Only keep samples where gold != -1 and pred != -1 ----------
        # For IO Prompt
        valid_mask_io = [(g != -1 and p != -1) for g, p in zip(gold_indices, io_indices)]
        valid_gold_io = [g for g, m in zip(gold_indices, valid_mask_io) if m]
        valid_io      = [p for p, m in zip(io_indices,  valid_mask_io) if m]

        # For CoT Prompt
        valid_mask_cot = [(g != -1 and p != -1) for g, p in zip(gold_indices, cot_indices)]
        valid_gold_cot = [g for g, m in zip(gold_indices, valid_mask_cot) if m]
        valid_cot      = [p for p, m in zip(cot_indices, valid_mask_cot) if m]

        # For ToT Prompt
        valid_mask_tot = [(g != -1 and p != -1) for g, p in zip(gold_indices, tot_indices)]
        valid_gold_tot = [g for g, m in zip(gold_indices, valid_mask_tot) if m]
        valid_tot      = [p for p, m in zip(tot_indices, valid_mask_tot) if m]

        # Calculation  (IO)
        if len(valid_gold_io) > 0:
            acc_io  = accuracy_score(valid_gold_io, valid_io)
            jaccard_io = jaccard_score(valid_gold_io, valid_io, average="macro")
        else:
            acc_io, jaccard_io = 0.0, 0.0

        # Calculation  (CoT)
        if len(valid_gold_cot) > 0:
            acc_cot  = accuracy_score(valid_gold_cot, valid_cot)
            jaccard_cot = jaccard_score(valid_gold_cot, valid_cot, average="macro")
        else:
            acc_cot, jaccard_cot = 0.0, 0.0

        # Calculation  (ToT)
        if len(valid_gold_tot) > 0:
            acc_tot  = accuracy_score(valid_gold_tot, valid_tot)
            jaccard_tot = jaccard_score(valid_gold_tot, valid_tot, average="macro")
        else:
            acc_tot, jaccard_tot = 0.0, 0.0
            
        final_gold_io  = valid_gold_io
        final_pred_io  = valid_io
        final_gold_cot = valid_gold_cot
        final_pred_cot = valid_cot
        final_gold_tot = valid_gold_tot
        final_pred_tot = valid_tot
        target_names   = ALL_PHASES    

    else:

        acc_io  = accuracy_score(gold_indices, io_indices)
        acc_cot = accuracy_score(gold_indices, cot_indices)
        acc_tot = accuracy_score(gold_indices, tot_indices)
    
        jaccard_io = jaccard_score(gold_indices,io_indices,average="macro")
        jaccard_cot = jaccard_score(gold_indices,cot_indices,average="macro")
        jaccard_tot = jaccard_score(gold_indices,tot_indices,average="macro")
        
        final_gold_io  = gold_indices
        final_pred_io  = io_indices
        final_gold_cot = gold_indices
        final_pred_cot = cot_indices
        final_gold_tot = gold_indices
        final_pred_tot = tot_indices
        target_names   = extended_phase_list
    
    print("\n===== Test Result (Accuracy) =====")
    print(f"IO Prompt  Accuracy: {acc_io:.3f}")
    print(f"CoT Prompt Accuracy: {acc_cot:.3f}")
    print(f"ToT Prompt Accuracy: {acc_tot:.3f}")
    
    print("\n===== Test Result (Jaccard Score) =====")
    print(f"IO Prompt Jaccard: {jaccard_io:.3f}")
    print(f"CoT Prompt Jaccard: {jaccard_cot:.3f}")
    print(f"ToT Prompt Jaccard: {jaccard_tot:.3f}")
    
    
    

    print("\n===== Classification Report (Precision、Recall、F1-score)(IO Prompt) =====")
    if len(final_gold_io) > 0 and len(final_pred_io) > 0:
        print(classification_report(final_gold_io, final_pred_io, target_names=target_names, zero_division=0))
    else:
        print("No valid samples for classification report (IO Prompt)")

    print("===== Classification Report (Precision、Recall、F1-score)(CoT Prompt) =====")
    if len(final_gold_cot) > 0 and len(final_pred_cot) > 0:
        print(classification_report(final_gold_cot, final_pred_cot, target_names=target_names, zero_division=0))
    else:
        print("No valid samples for classification report (CoT Prompt)")

    print("===== Classification Report (Precision、Recall、F1-score)(ToT Prompt) =====")
    if len(final_gold_tot) > 0 and len(final_pred_tot) > 0:
        print(classification_report(final_gold_tot, final_pred_tot, target_names=target_names, zero_division=0))
    else:
        print("No valid samples for classification report (ToT Prompt)")

    print("\n===== Confusion Matrix (IO Prompt) =====")
    if len(final_gold_io) > 0 and len(final_pred_io) > 0:
        print(confusion_matrix(final_gold_io, final_pred_io))
    else:
        print("No valid samples for confusion matrix(IO Prompt)")

    print("===== Confusion Matrix (CoT Prompt) =====")
    if len(final_gold_cot) > 0 and len(final_pred_cot) > 0:
        print(confusion_matrix(final_gold_cot, final_pred_cot))
    else:
        print("No valid samples for confusion matrix(CoT Prompt)")

    print("===== Confusion Matrix (ToT Prompt) =====")
    if len(final_gold_tot) > 0 and len(final_pred_tot) > 0:
        print(confusion_matrix(final_gold_tot, final_pred_tot))
    else:
        print("No valid samples for confusion matrix(ToT Prompt)")
        
        
    # Save Result
    df["Predict_IO"]  = results_io
    df["Predict_CoT"] = results_cot
    df["Predict_ToT"] = results_tot
    df.to_csv("prediction_results_hf.csv", index=False, encoding='utf-8')
    print("\nThe prediction results have been saved to prediction_results_hf.csv")


if __name__ == "__main__":
    """
    You can adjust sample_size based on hardware conditions and inference speed requirements.
    """
    args = parse_args()
    print_system_info()
    main_evaluation(
        folder_path=args.data_folder,
        model_dir=args.model_dir,
        sample_size=args.sample_size, 
        device=args.device,
        oov_as_other=args.oov_as_other,
        strict_match=args.strict_match
    )
