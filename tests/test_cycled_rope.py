"""
Unit tests for cycled/message-strided positional embeddings with speaker parity alignment.
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import compute_conversation_position_ids
from nanochat.engine import Engine, KVCache



class MockTokenizer:
    """Mock tokenizer providing the required special token IDs and interfaces for testing."""
    def __init__(self):
        self._special = {
            "<|bos|>": 1,
            "<|user_start|>": 2,
            "<|user_end|>": 3,
            "<|assistant_start|>": 4,
            "<|assistant_end|>": 5,
            "<|python_start|>": 6,
            "<|python_end|>": 7,
            "<|output_start|>": 8,
            "<|output_end|>": 9,
        }
        self.bos_token_id = 1

    def encode_special(self, text):
        return self._special.get(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, *args, **kwargs):
        # simple byte/char encoding shifted above special tokens
        return [ord(c) + 10 for c in text]

    def decode(self, ids):
        return "".join(chr(i - 10) for i in ids if i >= 10)

    def get_vocab_size(self):
        return 1000

    def compute_position_ids(self, token_ids, message_stride=512):
        return compute_conversation_position_ids(token_ids, self, message_stride)


def test_compute_conversation_position_ids_parity():
    tok = MockTokenizer()
    stride = 512

    # Conversation: BOS, user_start, text(3), user_end, asst_start, text(2), asst_end, user_start, text(2), user_end
    # Indices:
    # 0: BOS (pos 0)
    # 1: user_start (pos 1)
    # 2..4: user text (pos 2..4)
    # 5: user_end (pos 5)
    # 6: asst_start -> jumps to odd multiple 1*512 = 512
    # 7..8: asst text (pos 513..514)
    # 9: asst_end (pos 515)
    # 10: user_start -> jumps to even multiple 2*512 = 1024
    # 11..12: user text (pos 1025..1026)
    # 13: user_end (pos 1027)

    tokens = [
        tok.get_bos_token_id(), # 0: bos
        tok.encode_special("<|user_start|>"), # 1
        100, 101, 102, # 2, 3, 4
        tok.encode_special("<|user_end|>"), # 5
        tok.encode_special("<|assistant_start|>"), # 6
        200, 201, # 7, 8
        tok.encode_special("<|assistant_end|>"), # 9
        tok.encode_special("<|user_start|>"), # 10
        300, 301, # 11, 12
        tok.encode_special("<|user_end|>"), # 13
    ]

    positions = compute_conversation_position_ids(tokens, tok, message_stride=stride)

    assert positions[0] == 0  # BOS
    assert positions[1] == 1  # user_start
    assert positions[2:5] == [2, 3, 4] # user text
    assert positions[5] == 5  # user_end
    assert positions[6] == 512 # assistant_start (odd multiple 1*512)
    assert positions[7:9] == [513, 514] # assistant text
    assert positions[9] == 515 # assistant_end
    assert positions[10] == 1024 # second user turn (even multiple 2*512)
    assert positions[11:13] == [1025, 1026]
    assert positions[13] == 1027


def test_compute_conversation_position_ids_long_message_overflow():
    tok = MockTokenizer()
    stride = 512

    # Simulate a user message that has length > 512 (e.g. 600 tokens)
    tokens = [tok.get_bos_token_id(), tok.encode_special("<|user_start|>")]
    tokens += [100] * 600
    tokens.append(tok.encode_special("<|user_end|>"))
    tokens.append(tok.encode_special("<|assistant_start|>"))
    tokens += [200] * 10
    tokens.append(tok.encode_special("<|assistant_end|>"))

    positions = compute_conversation_position_ids(tokens, tok, message_stride=stride)

    # User message ends around pos 602
    # Next assistant turn must jump to the next odd multiple >= 603: 3 * 512 = 1536
    asst_start_idx = 2 + 600 + 1
    assert positions[asst_start_idx] == 3 * stride
    assert positions[asst_start_idx + 1] == 3 * stride + 1


def test_gpt_forward_position_ids_equivalence():
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        message_stride=0,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    device = model.get_device()
    B, T = 2, 32
    idx = torch.randint(0, 1000, (B, T), device=device)

    # 1. Forward with default contiguous positions
    out_default = model.forward(idx)

    # 2. Forward with explicit contiguous position_ids
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    out_explicit = model.forward(idx, position_ids=pos_ids)

    # Should be identical
    assert torch.allclose(out_default, out_explicit, atol=1e-5)


def test_gpt_forward_with_strided_position_ids():
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        message_stride=512,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    device = model.get_device()
    B, T = 2, 16
    idx = torch.randint(0, 1000, (B, T), device=device)

    # Create strided position IDs: first 8 tokens at [0..7], next 8 tokens at [512..519]
    pos = torch.cat([torch.arange(0, 8), torch.arange(512, 520)]).to(device)
    pos_ids = pos.unsqueeze(0).expand(B, -1)

    out_strided = model.forward(idx, position_ids=pos_ids)
    assert out_strided.shape == (B, T, 1000)
    assert not torch.isnan(out_strided).any()


def test_engine_generate_with_message_stride():
    config = GPTConfig(
        sequence_len=256,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        message_stride=64,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    tok = MockTokenizer()
    engine = Engine(model, tok)

    prompt = [
        tok.get_bos_token_id(),
        tok.encode_special("<|user_start|>"),
        50, 51, 52,
        tok.encode_special("<|user_end|>"),
        tok.encode_special("<|assistant_start|>"),
    ]

    results, masks = engine.generate_batch(prompt, num_samples=2, max_tokens=10, temperature=0.0)
    assert len(results) == 2
    assert len(results[0]) > len(prompt)


def test_compute_conversation_position_ids_with_python_tools():
    tok = MockTokenizer()
    stride = 512

    tokens = [
        tok.get_bos_token_id(), # 0: bos (pos 0)
        tok.encode_special("<|user_start|>"), # 1: user_start (pos 1)
        50, 51, # 2, 3: user text
        tok.encode_special("<|user_end|>"), # 4: user_end (pos 4)
        tok.encode_special("<|assistant_start|>"), # 5: asst_start (odd multiple 1*512 = 512)
        tok.encode_special("<|python_start|>"), # 6: python_start (within asst turn -> odd multiple >= pos: stays odd 513)
        60, 61, # 7, 8: python code (pos 514, 515)
        tok.encode_special("<|python_end|>"), # 9: python_end (pos 516)
        tok.encode_special("<|output_start|>"), # 10: output_start -> jumps to next even multiple 2*512 = 1024
        70, 71, # 11, 12: output text (pos 1025, 1026)
        tok.encode_special("<|output_end|>"), # 13: output_end (pos 1027)
        tok.encode_special("<|assistant_end|>"), # 14: asst_end (pos 1028)
    ]

    positions = compute_conversation_position_ids(tokens, tok, message_stride=stride)
    assert positions[0] == 0
    assert positions[5] == 512 # assistant_start
    assert positions[10] == 1024 # output_start (even parity)

