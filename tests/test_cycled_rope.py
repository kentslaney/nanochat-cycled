"""
Test cycled RoPE: message-strided virtual positions and the rotary path that consumes them.

python -m pytest tests/test_cycled_rope.py -v

Three layers are covered:
1. The position scheme itself (pure Python, no torch): parity, overflow, tool boundaries.
2. The model: equivalence with the contiguous path, and correctness of the custom
   autograd Function that recomputes cos/sin on backward.
3. The Engine: prefill and decode reproduce the same positions the data loader would.
"""

from dataclasses import dataclass

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb, apply_rotary_emb_strided, rotary_cos_sin
from nanochat.tokenizer import (
    ASSISTANT_PARITY,
    USER_PARITY,
    PositionTracker,
    RustBPETokenizer,
    SPECIAL_TOKENS,
    compute_conversation_position_ids,
    conversation_boundaries,
)
from nanochat.engine import Engine

STRIDE = 512


# -----------------------------------------------------------------------------
# Layer 1: the position scheme (pure Python)
#
# Synthetic token ids keep these tests independent of any particular BPE vocab.
BOS, USER_START, USER_END = 100, 101, 102
ASST_START, ASST_END = 103, 104
PY_START, PY_END = 105, 106
OUT_START, OUT_END = 107, 108
WORD = 999  # any ordinary, non-boundary token

ALIGN_AT = {
    BOS: USER_PARITY,
    USER_START: USER_PARITY,
    OUT_START: USER_PARITY,
    ASST_START: ASSISTANT_PARITY,
    PY_START: ASSISTANT_PARITY,
}
ALIGN_AFTER = {OUT_END: ASSISTANT_PARITY}


def positions(ids, stride=STRIDE):
    return compute_conversation_position_ids(ids, stride, ALIGN_AT, ALIGN_AFTER)


def slot(pos, stride=STRIDE):
    """The stride multiple pos sits on. Asserts pos opens that slot exactly."""
    assert pos % stride == 0, f"position {pos} is not a multiple of the stride {stride}"
    return pos // stride


def region(pos, stride=STRIDE):
    """Which speaker's half of the position space pos falls in (0 = user, 1 = assistant).

    Unlike slot(), this does not require pos to be stride-aligned: a message continues
    contiguously inside the slot it opened.
    """
    return (pos // stride) % 2


def test_stride_zero_is_contiguous():
    ids = [BOS, USER_START, WORD, USER_END, ASST_START, WORD, ASST_END]
    assert compute_conversation_position_ids(ids, 0, ALIGN_AT, ALIGN_AFTER) == list(range(len(ids)))


def test_documented_example():
    """The exact worked example from the design: BOS shares slot 0 with the first user turn."""
    ids = [BOS, USER_START, WORD, USER_END, ASST_START, WORD, ASST_END, USER_START, WORD]
    assert positions(ids) == [0, 1, 2, 3, 512, 513, 514, 1024, 1025]


def test_speaker_parity():
    """User turns land on even multiples of the stride, assistant turns on odd ones."""
    ids = []
    for _ in range(4):  # four full exchanges
        ids += [USER_START, WORD, WORD, USER_END, ASST_START, WORD, WORD, ASST_END]
    ids = [BOS] + ids
    pos = positions(ids)

    assert pos[0] == 0, "BOS starts the document at position 0"
    for token, position in zip(ids, pos):
        if token == ASST_START:
            assert slot(position) % 2 == 1, f"assistant turn at {position} is not on an odd multiple"
    # every user turn except the very first (which shares slot 0 with BOS) is stride-aligned
    user_starts = [p for t, p in zip(ids, pos) if t == USER_START]
    assert user_starts[0] == 1, "the first user turn continues contiguously after BOS"
    for position in user_starts[1:]:
        assert slot(position) % 2 == 0, f"user turn at {position} is not on an even multiple"
    # positions are strictly increasing, so causality (and the KV cache) is never violated
    assert all(b > a for a, b in zip(pos, pos[1:]))


def test_position_is_independent_of_earlier_message_lengths():
    """The whole point: message N starts in the same place no matter how long messages 0..N-1 are."""
    def render(user_len, asst_len):
        ids = [BOS, USER_START] + [WORD] * user_len + [USER_END]
        ids += [ASST_START] + [WORD] * asst_len + [ASST_END]
        ids += [USER_START, WORD, USER_END]  # the message we care about
        return ids

    short, long = render(3, 4), render(40, 90)
    pos_short, pos_long = positions(short), positions(long)
    # the trailing user turn starts at the same virtual position in both renderings
    assert pos_short[-3:] == pos_long[-3:] == [1024, 1025, 1026]


def test_long_message_overflow_skips_to_next_aligned_slot():
    """A message longer than the stride must not leave the next one starting mid-slot."""
    # a user turn of 600 tokens overflows slot 0 and spills into slot 1 (an assistant slot)
    ids = [BOS, USER_START] + [WORD] * 600 + [USER_END, ASST_START, WORD]
    pos = positions(ids)
    asst = pos[ids.index(ASST_START)]
    assert pos[-3] > STRIDE, "precondition: the user turn really did overflow its slot"
    assert slot(asst) == 3, f"assistant should skip to slot 3, got position {asst}"

    # and a turn spanning more than two slots skips correspondingly further
    ids = [BOS, USER_START] + [WORD] * 1100 + [USER_END, ASST_START, WORD]
    pos = positions(ids)
    assert slot(pos[ids.index(ASST_START)]) == 3


def test_tool_use_alignment():
    """Tool code is the assistant speaking (odd); tool output is fed to it (even)."""
    ids = [
        BOS, USER_START, WORD, USER_END,
        ASST_START, WORD,
        PY_START, WORD, PY_END,
        OUT_START, WORD, OUT_END,
        WORD, ASST_END,  # the assistant resumes after seeing the tool output
    ]
    pos = positions(ids)
    by_token = dict(zip(ids, pos))  # every boundary token appears once here

    assert slot(by_token[ASST_START]) % 2 == 1
    # tool code is the same speaker as the text before it, so it simply continues in that
    # odd slot - it only jumps if the assistant's preamble overflowed (see the test below)
    assert region(by_token[PY_START]) == 1, "tool code is written by the assistant"
    assert by_token[PY_START] == by_token[ASST_START] + 2
    # tool output is fed *to* the assistant, so it opens a fresh even slot
    assert slot(by_token[OUT_START]) % 2 == 0, "tool output is not on an even multiple"
    # <|output_end|> closes the output block, so it stays in the output's even slot ...
    assert by_token[OUT_END] == by_token[OUT_START] + 2
    # ... and the assistant's next token opens a fresh odd slot, so what it says next does
    # not depend on how long the tool output happened to be
    resumed = pos[ids.index(OUT_END) + 1]
    assert slot(resumed) % 2 == 1
    assert resumed > by_token[OUT_END]


def test_tool_code_jumps_when_the_preamble_overflows():
    """If the assistant's text overflows its slot, the tool call still lands on an odd one."""
    ids = [BOS, USER_START, WORD, USER_END, ASST_START] + [WORD] * 600 + [PY_START, WORD, PY_END]
    pos = positions(ids)
    assert slot(pos[ids.index(PY_START)]) % 2 == 1


def test_tracker_is_incremental_and_clonable():
    """Feeding tokens one at a time matches doing it in bulk, and clones diverge cleanly."""
    ids = [BOS, USER_START, WORD, USER_END, ASST_START, WORD]
    tracker = PositionTracker(STRIDE, ALIGN_AT, ALIGN_AFTER)
    assert [tracker.step(t) for t in ids] == positions(ids)

    # a clone continues from the same state but is independent afterwards
    clone = tracker.clone()
    assert tracker.step(ASST_END) == clone.step(ASST_END)
    # now they diverge: one keeps talking, the other starts a new user turn
    assert tracker.step(WORD) != clone.step(USER_START)
    assert slot(clone.pos - 1) % 2 == 0


def test_pending_alignment_survives_a_clone():
    """<|output_end|> defers its realignment to the next token; clones must carry that."""
    tracker = PositionTracker(STRIDE, ALIGN_AT, ALIGN_AFTER)
    tracker.extend([BOS, USER_START, USER_END, ASST_START, OUT_START, WORD, OUT_END])
    clone = tracker.clone()
    assert slot(clone.step(WORD)) % 2 == 1, "the deferred odd realignment was lost by clone()"


# -----------------------------------------------------------------------------
# Layer 1b: the same scheme, driven through the real tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world, hello tokenizer, hello hello hello",
    "def f(x):\n    return x + 1\n",
] * 8


@pytest.fixture(scope="module")
def tokenizer():
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), 256 + len(SPECIAL_TOKENS) + 35)


def test_render_conversation_returns_positions(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hello back"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "later"},
    ]}
    ids, mask, pos = tokenizer.render_conversation(conversation, message_stride=STRIDE, return_positions=True)
    assert len(ids) == len(mask) == len(pos)
    # unchanged behaviour when positions are not requested
    assert (ids, mask) == tokenizer.render_conversation(conversation)

    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    user_start = tokenizer.encode_special("<|user_start|>")
    for token, position in zip(ids, pos):
        if token == assistant_start:
            assert slot(position) % 2 == 1
    assert slot([p for t, p in zip(ids, pos) if t == user_start][1]) % 2 == 0


def test_render_conversation_stride_zero_is_contiguous(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]}
    ids, _, pos = tokenizer.render_conversation(conversation, message_stride=0, return_positions=True)
    assert pos == list(range(len(ids)))


def test_render_conversation_positions_respect_truncation(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hello " * 50},
        {"role": "assistant", "content": "world " * 50},
    ]}
    ids, mask, pos = tokenizer.render_conversation(
        conversation, max_tokens=16, message_stride=STRIDE, return_positions=True)
    assert len(ids) == len(mask) == len(pos) == 16


def test_conversation_boundaries_cover_the_chat_format(tokenizer):
    align_at, align_after = conversation_boundaries(tokenizer)
    assert align_at[tokenizer.encode_special("<|assistant_start|>")] == ASSISTANT_PARITY
    assert align_at[tokenizer.encode_special("<|python_start|>")] == ASSISTANT_PARITY
    assert align_at[tokenizer.encode_special("<|user_start|>")] == USER_PARITY
    assert align_at[tokenizer.encode_special("<|output_start|>")] == USER_PARITY
    assert align_at[tokenizer.get_bos_token_id()] == USER_PARITY
    assert align_after == {tokenizer.encode_special("<|output_end|>"): ASSISTANT_PARITY}
    # <|user_end|>, <|assistant_end|>, <|python_end|> close their blocks and never realign
    for name in ("<|user_end|>", "<|assistant_end|>", "<|python_end|>"):
        assert tokenizer.encode_special(name) not in align_at
        assert tokenizer.encode_special(name) not in align_after


# -----------------------------------------------------------------------------
# Layer 2: the rotary implementation


def reference_rotary(q, k, position_ids, inv_freq):
    """Plain autograd version of apply_rotary_emb_strided, used as ground truth."""
    freqs = position_ids.to(inv_freq.dtype).unsqueeze(-1).unsqueeze(2) * inv_freq
    cos, sin = freqs.cos().to(q.dtype), freqs.sin().to(q.dtype)
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def test_rotary_cos_sin_shape_and_values():
    inv_freq = torch.tensor([1.0, 0.5, 0.25])
    position_ids = torch.tensor([[0, 1], [512, 513]])
    cos, sin = rotary_cos_sin(position_ids, inv_freq, torch.float32)
    assert cos.shape == sin.shape == (2, 2, 1, 3)  # (B, T, 1, head_dim/2)
    assert torch.allclose(cos[0, 0, 0], torch.ones(3))  # position 0 => angle 0
    assert torch.allclose(sin[1, 0, 0], torch.sin(512 * inv_freq), atol=1e-6)


def test_strided_rotary_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    B, T, H, D = 2, 6, 3, 8
    inv_freq = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    position_ids = torch.tensor([[0, 1, 2, 512, 513, 1024], [0, 1, 512, 513, 514, 1536]])

    q = torch.randn(B, T, H, D, dtype=torch.float64)
    k = torch.randn(B, T, H, D, dtype=torch.float64)
    grads = []
    for fn in (reference_rotary, apply_rotary_emb_strided):
        qi, ki = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        q_out, k_out = fn(qi, ki, position_ids, inv_freq)
        if fn is reference_rotary:
            expected = (q_out.detach(), k_out.detach())
        else:
            assert torch.allclose(q_out, expected[0]) and torch.allclose(k_out, expected[1])
        # a non-symmetric scalar objective, so wrong gradients cannot cancel out
        weights = torch.linspace(0.1, 1.0, T, dtype=torch.float64).view(1, T, 1, 1)
        loss = (q_out * weights).sum() + (k_out * weights.flip(1)).sum()
        loss.backward()
        grads.append((qi.grad, ki.grad))
    assert torch.allclose(grads[0][0], grads[1][0])
    assert torch.allclose(grads[0][1], grads[1][1])


def test_strided_rotary_gradcheck():
    """Numerical gradient check of the hand-written backward."""
    B, T, H, D = 2, 4, 2, 6
    inv_freq = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    position_ids = torch.tensor([[0, 1, 512, 513], [0, 512, 1024, 1025]])
    q = torch.randn(B, T, H, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: apply_rotary_emb_strided(a, b, position_ids, inv_freq), (q, k))


def test_strided_rotary_preserves_norms_and_relative_angles():
    """It is a rotation: lengths are preserved and q.k depends only on the position delta."""
    torch.manual_seed(0)
    D = 8
    inv_freq = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    q = torch.randn(1, 2, 1, D)
    k = torch.randn(1, 2, 1, D)

    near = torch.tensor([[0, 1]])          # delta of 1 inside a message
    far = torch.tensor([[512, 513]])       # same delta, but a whole slot later
    q_near, k_near = apply_rotary_emb_strided(q, k, near, inv_freq)
    q_far, k_far = apply_rotary_emb_strided(q, k, far, inv_freq)

    assert torch.allclose(q_near.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    dot_near = (q_near[0, 1] * k_near[0, 0]).sum()
    dot_far = (q_far[0, 1] * k_far[0, 0]).sum()
    assert torch.allclose(dot_near, dot_far, atol=1e-3), "RoPE must only see the position difference"


# -----------------------------------------------------------------------------
# Layer 2b: the model


def tiny_model(message_stride=0, sequence_len=64):
    config = GPTConfig(
        sequence_len=sequence_len, vocab_size=64, n_layer=2, n_head=2,
        n_kv_head=2, n_embd=32, window_pattern="L", message_stride=message_stride,
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.eval()
    return model


def test_config_default_is_off():
    assert GPTConfig().message_stride == 0
    assert tiny_model().config.message_stride == 0


def test_inv_freq_buffer():
    model = tiny_model()
    head_dim = model.config.n_embd // model.config.n_head
    assert model.inv_freq.shape == (head_dim // 2,)
    assert model.inv_freq.dtype == torch.float32
    assert "inv_freq" not in model.state_dict(), "inv_freq is derived, it must not be checkpointed"
    # the buffer and the precomputed table must describe the same rotation
    cos, _ = rotary_cos_sin(torch.arange(8).unsqueeze(0), model.inv_freq, model.cos.dtype)
    assert torch.allclose(cos[0], model.cos[0, :8], atol=1e-5)


def test_contiguous_position_ids_match_the_default_path():
    """position_ids=arange(T) must reproduce the precomputed-table path exactly."""
    torch.manual_seed(0)
    model = tiny_model()
    B, T = 2, 16
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    position_ids = torch.arange(T).unsqueeze(0).expand(B, T)
    with torch.no_grad():
        default = model(idx)
        explicit = model(idx, position_ids=position_ids)
    assert torch.allclose(default, explicit, atol=1e-5)


def test_strided_positions_produce_finite_outputs_and_gradients():
    torch.manual_seed(0)
    model = tiny_model(message_stride=STRIDE)
    model.train()
    B, T = 2, 12
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    targets = torch.randint(0, model.config.vocab_size, (B, T))
    # positions well beyond rotary_seq_len: the strided path must not touch the table
    position_ids = torch.tensor([
        [0, 1, 2, 512, 513, 514, 1024, 1025, 1536, 1537, 2048, 2049],
        [0, 1, 512, 513, 1024, 1025, 1026, 1536, 2048, 2049, 2560, 2561],
    ])
    assert position_ids.max() > model.rotary_seq_len

    loss = model(idx, targets, position_ids=position_ids)
    assert torch.isfinite(loss), "strided positions produced a non-finite loss"
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_positions_change_the_output():
    """Sanity: the strided positions are actually reaching the attention, not being ignored."""
    torch.manual_seed(0)
    model = tiny_model(message_stride=STRIDE)
    B, T = 1, 8
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    with torch.no_grad():
        contiguous = model(idx, position_ids=torch.arange(T).unsqueeze(0))
        strided = model(idx, position_ids=torch.tensor([[0, 1, 2, 512, 513, 1024, 1025, 1026]]))
    assert not torch.allclose(contiguous, strided, atol=1e-4)


def test_forward_rejects_mismatched_position_ids():
    model = tiny_model(message_stride=STRIDE)
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    with pytest.raises(AssertionError):
        model(idx, position_ids=torch.arange(8))  # missing the batch dimension


# -----------------------------------------------------------------------------
# Layer 3: the Engine


class FakeChatTokenizer:
    """Just enough tokenizer for the Engine: the chat special tokens plus byte encoding."""
    SPECIALS = {
        "<|bos|>": 0, "<|user_start|>": 1, "<|user_end|>": 2,
        "<|assistant_start|>": 3, "<|assistant_end|>": 4,
        "<|python_start|>": 5, "<|python_end|>": 6,
        "<|output_start|>": 7, "<|output_end|>": 8,
    }
    OFFSET = 9  # ordinary tokens live above the specials

    def encode_special(self, text):
        return self.SPECIALS[text]

    def get_bos_token_id(self):
        return self.SPECIALS["<|bos|>"]

    def encode(self, text, prepend=None):
        ids = [b + self.OFFSET for b in text.encode("utf-8")]
        return ([prepend] + ids) if prepend is not None else ids

    def decode(self, ids):
        return bytes(i - self.OFFSET for i in ids if i >= self.OFFSET).decode("utf-8", errors="replace")


@dataclass
class ScriptedConfig:
    n_kv_head: int = 2
    n_head: int = 2
    n_embd: int = 32
    n_layer: int = 2
    sequence_len: int = 4096
    message_stride: int = STRIDE


class ScriptedModel:
    """Emits a fixed token sequence (argmax over one-hot logits) and records what it was fed."""

    def __init__(self, script, vocab_size=280, message_stride=STRIDE):
        self.script = list(script)
        self.vocab_size = vocab_size
        self.config = ScriptedConfig(message_stride=message_stride)
        self.calls = 0
        self.seen_positions = []  # position_ids of each forward, as nested lists (or None)

    def get_device(self):
        return torch.device("cpu")

    def forward(self, ids, kv_cache=None, position_ids=None):
        B, T = ids.shape
        self.seen_positions.append(None if position_ids is None else position_ids.tolist())
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = torch.zeros(B, T, self.vocab_size)
        if self.calls < len(self.script):
            logits[:, -1, self.script[self.calls]] = 100.0
        self.calls += 1
        return logits


def engine_positions(prompt, generated, message_stride=STRIDE):
    """The positions the whole sequence should get, computed independently of the Engine."""
    tokenizer = FakeChatTokenizer()
    tracker = PositionTracker(message_stride, *conversation_boundaries(tokenizer))
    return tracker.extend(list(prompt) + list(generated))


def test_engine_is_off_for_models_without_a_stride():
    model = ScriptedModel([10, 11], message_stride=0)
    engine = Engine(model, FakeChatTokenizer())
    assert not engine.strided_positions
    list(engine.generate([0, 1, 10, 2], num_samples=1, max_tokens=2, temperature=0.0))
    assert all(seen is None for seen in model.seen_positions), "no positions should be passed"


def test_engine_prefill_and_decode_positions():
    tokenizer = FakeChatTokenizer()
    prompt = [0, 1] + tokenizer.encode("hi") + [2, 3]  # bos, user turn, then assistant_start
    generated = tokenizer.encode("ok") + [4]           # "ok" then <|assistant_end|>
    model = ScriptedModel(generated)
    engine = Engine(model, tokenizer)

    emitted = [column[0] for column, _ in engine.generate(prompt, num_samples=1, max_tokens=3, temperature=0.0)]
    assert emitted == generated, "precondition: the scripted model drove the generation"

    expected = engine_positions(prompt, generated)
    assert model.seen_positions[0] == [expected[:len(prompt)]], "prefill positions are wrong"
    # after the prefill there is one decode forward per generated token
    assert len(model.seen_positions) == 1 + len(generated)
    for i, seen in enumerate(model.seen_positions[1:]):
        assert seen == [[expected[len(prompt) + i]]], f"decode step {i} position is wrong"

    # the assistant turn really is on an odd slot, and the response continues inside it
    assert slot(expected[len(prompt) - 1]) % 2 == 1
    assert expected[len(prompt)] == expected[len(prompt) - 1] + 1


def test_engine_rows_track_positions_independently():
    """Rows that emit different tokens must not share position state."""
    tokenizer = FakeChatTokenizer()
    prompt = [0, 1] + tokenizer.encode("hi") + [2]
    # every row is fed the same script here, but each row must still own its tracker
    model = ScriptedModel([3, 10, 4])
    engine = Engine(model, tokenizer)
    num_samples = 4
    list(engine.generate(prompt, num_samples=num_samples, max_tokens=3, temperature=0.0))

    expected = engine_positions(prompt, [3, 10, 4])
    for i, seen in enumerate(model.seen_positions[1:]):
        assert seen == [[expected[len(prompt) + i]]] * num_samples

    # <|assistant_start|> was sampled during decode, so it must have opened a new odd slot
    assert slot(expected[len(prompt)]) % 2 == 1


def test_engine_tool_output_positions():
    """Forced tool-output tokens realign exactly like tokenized ones."""
    tokenizer = FakeChatTokenizer()
    prompt = [0, 1] + tokenizer.encode("2+2") + [2, 3]
    # assistant writes <|python_start|>2+2<|python_end|>; the engine then forces the output block
    script = [5] + tokenizer.encode("2+2") + [6] + [4] * 8
    model = ScriptedModel(script)
    engine = Engine(model, tokenizer)

    emitted = [column[0] for column, _ in engine.generate(prompt, num_samples=1, max_tokens=12, temperature=0.0)]
    output_start = tokenizer.encode_special("<|output_start|>")
    assert output_start in emitted, "precondition: the calculator result was force-injected"

    expected = engine_positions(prompt, emitted)
    seen = [model.seen_positions[0][0][-1]] + [s[0][0] for s in model.seen_positions[1:]]
    assert seen == expected[len(prompt) - 1:len(prompt) - 1 + len(seen)]
    # the tool call continues the assistant's odd region, the forced output block opens an even slot
    assert region(expected[len(prompt) + emitted.index(5)]) == 1
    assert slot(expected[len(prompt) + emitted.index(output_start)]) % 2 == 0
    # and the assistant's reply after the output starts a fresh odd slot
    assert slot(expected[len(prompt) + emitted.index(tokenizer.encode_special("<|output_end|>")) + 1]) % 2 == 1
