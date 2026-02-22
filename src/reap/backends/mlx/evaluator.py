"""MLX evaluator for REAP framework (replaces vLLM)."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    from mlx_lm import generate, load
    from mlx_lm.utils import load_config
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logger.warning("mlx-lm not available. Install with: pip install mlx-lm")


class MlxEvaluator:
    """
    Evaluation pipeline using MLX-LM.

    This replaces vLLM for Apple Silicon evaluation.
    MLX-LM provides direct inference without needing a server.
    """

    def __init__(self, model_path: str, model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to MLX model
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        if not MLX_LM_AVAILABLE:
            raise ImportError("mlx-lm is required for MlxEvaluator")

        self.model_path = model_path
        self._model = model
        self._tokenizer = tokenizer

        if model is None or tokenizer is None:
            self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        self._model, self._tokenizer = load(self.model_path)

    def generate(self, prompts: List[str], max_tokens: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 20, **kwargs) -> List[str]:
        """
        Generate responses for prompts.

        Equivalent to vLLM's generate function.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            **kwargs: Additional generation args

        Returns:
            List of generated responses
        """
        responses = []

        for prompt in prompts:
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )
            responses.append(response)

        return responses

    def generate_greedy(self, prompts: List[str], max_tokens: int = 100,
                       **kwargs) -> List[str]:
        """
        Generate using greedy decoding (temperature=0).

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation args

        Returns:
            List of generated responses
        """
        return self.generate(prompts, max_tokens=max_tokens, temperature=0.0, **kwargs)

    def benchmark_throughput(self, batch_size: int = 8,
                            seq_len: int = 512,
                            num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model throughput (tokens/second).

        Replaces vLLM's benchmarking.

        Args:
            batch_size: Batch size for benchmarking
            seq_len: Sequence length
            num_iterations: Number of iterations

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark: batch={batch_size}, seq_len={seq_len}")

        # Generate random input
        vocab_size = self._tokenizer.vocab_size if hasattr(self._tokenizer, 'vocab_size') else 32000
        input_ids = mx.randint(0, vocab_size, (batch_size, seq_len))

        # Warmup
        logger.info("Warming up...")
        for _ in range(3):
            _ = self._model(input_ids)
        mx.eval(self._model.parameters())

        # Benchmark
        logger.info(f"Running {num_iterations} iterations...")
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self._model(input_ids)
            mx.eval(self._model.parameters())

        end_time = time.time()

        total_tokens = batch_size * seq_len * num_iterations
        elapsed_time = end_time - start_time

        throughput = total_tokens / elapsed_time

        logger.info(f"Throughput: {throughput:.2f} tokens/sec")

        return {
            "tokens_per_second": throughput,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "num_iterations": num_iterations,
            "elapsed_time": elapsed_time,
            "total_tokens": total_tokens,
            "backend": "mlx",
        }

    def evaluate_perplexity(self, test_data: List[str],
                           max_length: int = 2048) -> float:
        """
        Evaluate model perplexity on test data.

        Args:
            test_data: List of text strings
            max_length: Maximum sequence length

        Returns:
            Perplexity score
        """
        import mlx.nn as nn

        total_loss = 0.0
        total_tokens = 0

        for text in test_data:
            tokens = self._tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            # Truncate if needed
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = mx.array(tokens[:-1]).reshape(1, -1)
            target_ids = mx.array(tokens[1:]).reshape(1, -1)

            # Forward pass
            logits = self._model(input_ids)

            # Calculate loss
            loss = nn.losses.cross_entropy(logits, target_ids)
            total_loss += loss.item() * (len(tokens) - 1)
            total_tokens += len(tokens) - 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = mx.exp(mx.array(avg_loss)).item()

        logger.info(f"Perplexity: {perplexity:.2f}")

        return perplexity

    def memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        MLX on Apple Silicon uses unified memory.
        """
        param_memory = 0

        for param in self._model.parameters():
            if hasattr(param, 'size') and hasattr(param, 'dtype'):
                param_memory += param.size * param.dtype.size

        return {
            "parameter_memory_mb": param_memory / (1024 * 1024),
            "backend": "mlx",
            "device": "metal" if mx.metal.is_available() else "cpu",
        }

    @property
    def model(self) -> Any:
        """Get the model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer


class MlxLMEvalAdapter:
    """
    Adapter to provide lm-eval compatible interface for MLX.

    This allows using MLX models with lm-eval tasks.
    """

    def __init__(self, model_path: str):
        """
        Initialize adapter.

        Args:
            model_path: Path to MLX model
        """
        self.evaluator = MlxEvaluator(model_path)
        self.model = self.evaluator.model
        self.tokenizer = self.evaluator.tokenizer

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000

    def generate_until(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Generate until stopping condition.

        Compatible with lm-eval interface.

        Args:
            requests: List of generation requests

        Returns:
            List of generated strings
        """
        responses = []

        for request in requests:
            prompt = request.get('prompt', '')
            max_tokens = request.get('max_gen_toks', 100)
            until = request.get('until', None)

            response = self.evaluator.generate(
                [prompt],
                max_tokens=max_tokens,
                temperature=0.0  # Greedy for eval
            )[0]

            # Trim to stopping condition
            if until:
                for stop_str in until if isinstance(until, list) else [until]:
                    if stop_str in response:
                        response = response.split(stop_str)[0]

            responses.append(response)

        return responses

    def loglikelihood(self, requests: List[Dict[str, Any]]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for prompts.

        Compatible with lm-eval interface.

        Args:
            requests: List of {prompt, continuation} pairs

        Returns:
            List of (logprob, is_greedy) tuples
        """
        import mlx.nn as nn

        results = []

        for request in requests:
            prompt = request['prompt']
            continuation = request.get('continuation', '')

            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt)
            cont_tokens = self.tokenizer.encode(continuation)

            # Get logits for prompt
            input_ids = mx.array(prompt_tokens).reshape(1, -1)
            logits = self.model(input_ids)

            # Get log probs for continuation tokens
            log_probs = []
            for i, token in enumerate(cont_tokens):
                next_token_logits = logits[0, len(prompt_tokens) + i - 1]
                next_token_probs = mx.softmax(next_token_logits, axis=-1)
                log_prob = mx.log(next_token_probs[token]).item()
                log_probs.append(log_prob)

            # Sum log probs
            total_logprob = sum(log_probs)

            # Check if greedy
            is_greedy = True
            for i, token in enumerate(cont_tokens):
                next_token_logits = logits[0, len(prompt_tokens) + i - 1]
                greedy_token = int(mx.argmax(next_token_logits, axis=-1).item())
                if greedy_token != token:
                    is_greedy = False
                    break

            results.append((total_logprob, is_greedy))

        return results
