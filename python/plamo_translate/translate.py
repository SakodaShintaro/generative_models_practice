"""Local translation with PLaMo 3 Translate (transformers backend).

Examples:
    uv run python plamo_translate/translate.py --device cuda:1 --target_lang ja --text "Hello"
    uv run python plamo_translate/translate.py --device cuda:1 --target_lang en  # interactive
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pfnet/plamo-3-translate-8b")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--target_lang", type=str, default="ja")
    parser.add_argument("--form", type=str, default="none", choices=["none", "plain", "polite"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--text", type=str, default="")
    return parser.parse_args()


def build_template_kwargs(target_lang: str, form: str) -> dict:
    kwargs = {"target_lang": target_lang}
    if form != "none":
        assert target_lang == "ja", "--form is only valid when --target_lang is ja"
        kwargs["form"] = form
    return kwargs


@torch.inference_mode()
def translate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    template_kwargs: dict,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(model.device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    template_kwargs = build_template_kwargs(args.target_lang, args.form)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    model.eval()

    if args.text:
        print(translate(model, tokenizer, args.text, template_kwargs, args.max_new_tokens))
        return

    print(f"Interactive mode (target={args.target_lang}). Empty line to translate, Ctrl-D to exit.")
    while True:
        lines = []
        for line in sys.stdin:
            if line.strip() == "":
                break
            lines.append(line.rstrip("\n"))
        else:
            if not lines:
                return
        if not lines:
            continue
        start = time.perf_counter()
        result = translate(model, tokenizer, "\n".join(lines), template_kwargs, args.max_new_tokens)
        elapsed = time.perf_counter() - start
        print(f"--- ({elapsed:.1f}s)\n{result}\n")


if __name__ == "__main__":
    main()
