import evals
import evals.record
from evals.api import CompletionFn
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

class MemeEval(evals.Eval):
    """
    Custom evaluation for:
    1. Retrieval coverage of required meme(s)
    2. Style/tone scoring of model outputs
    """

    def __init__(self, completion_fns: list[CompletionFn], *args, **kwargs):
        super().__init__(completion_fns, *args, **kwargs)

    def run_sample(self, sample, rng):
        query = sample["query"]
        required_memes = sample.get("required_memes", [])
        retrieved = sample["retrieved"]   # list of meme_names from retriever
        model_output = sample["model_output"]

        # 1. Retrieval check - normalized score (0-1)
        if required_memes:
            retrieval_score = sum(1 for m in required_memes if m in retrieved) / len(required_memes)
        else:
            retrieval_score = 1.0  # Perfect score if no specific memes required

        # 2. Style/tone check (0–5)
        # Use LLM-as-judge to rate style
        style_prompt = f"""
            你是一个评论风格评估员。请判断下面文本是否自然、随意、幽默、有网络玩梗的感觉。
            文本: {model_output}

            请给出一个分数 (0-5)，5表示完全符合要求，0表示完全不符合。
            只输出数字。
            """
        try:
            style_eval = self.completion_fns[0](
                prompt=style_prompt,
                temperature=0,
                max_tokens=5,
            )
            # Extract first completion and parse score
            completion_text = style_eval.get_completions()[0]
            # Extract first digit found in the response
            digits = "".join(c for c in completion_text if c.isdigit())
            if digits:
                style_score = min(int(digits[0]), 5)  # Ensure score is 0-5
            else:
                style_score = 0
        except Exception as e:
            print(f"Style evaluation failed: {e}")
            style_score = 0

        # Log metrics
        evals.record.record_metrics(
            retrieval_score=retrieval_score,
            style_score=style_score,
        )

        return {"retrieval_score": retrieval_score, "style_score": style_score}
