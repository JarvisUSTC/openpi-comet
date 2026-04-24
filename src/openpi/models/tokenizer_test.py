import numpy as np

from openpi.models import tokenizer as _tokenizer


class _FakeSentencePiece:
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [ord(ch) % 97 + 2 for ch in text]
        if add_bos:
            tokens = [1] + tokens
        if add_eos:
            tokens = tokens + [2]
        return tokens

    def decode(self, tokens):
        return "".join(chr((int(t) - 2) % 97 + 30) for t in tokens if int(t) > 1)

    def vocab_size(self):
        return 4096


class _FakeFastProcessor:
    def __call__(self, actions):
        horizon, action_dim = actions.shape[1:]
        return [list(range(horizon * action_dim))]

    def decode(self, tokens, *, time_horizon: int, action_dim: int):
        return [np.zeros((time_horizon, action_dim), dtype=np.float32)]


def _make_fast_tokenizer(max_len: int = 256) -> _tokenizer.FASTTokenizer:
    tokenizer = _tokenizer.FASTTokenizer.__new__(_tokenizer.FASTTokenizer)
    tokenizer._max_len = max_len
    tokenizer._paligemma_tokenizer = _FakeSentencePiece()
    tokenizer._fast_tokenizer = _FakeFastProcessor()
    tokenizer._fast_skip_tokens = 128
    return tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _make_fast_tokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks, flow_tokens, flow_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)
    assert flow_tokens.shape == (256,)
    assert flow_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_fast_tokenizer_answer_supervision():
    prompt = "What is in the image?"
    state = np.zeros((4,), dtype=np.float32)
    tokenizer = _make_fast_tokenizer(max_len=64)
    tokens, token_masks, ar_masks, loss_masks, flow_tokens, flow_masks = tokenizer.tokenize(
        prompt,
        state,
        None,
        answer="red cube",
    )

    assert tokens.shape == (64,)
    assert flow_tokens.shape == (64,)
    assert flow_masks.sum() < token_masks.sum()
    assert loss_masks.any()
    assert ar_masks[flow_masks.sum()] == 1


def test_fast_tokenizer_no_state_prefix():
    prompt = "What is in the image?"
    tokenizer = _make_fast_tokenizer(max_len=64)

    prefix_tokens = tokenizer._tokenize_prefix(prompt, None)
    expected = tokenizer._paligemma_tokenizer.encode(
        "Task: what is in the image?, State: <no_state>;\n",
        add_bos=True,
    )

    assert prefix_tokens == expected


def test_fast_tokenizer_truncation_preserves_answer_postfix():
    tokenizer = _make_fast_tokenizer(max_len=64)
    prompt = "x" * 500  # Force prefix overflow
    postfix_tokens = tokenizer._tokenize_postfix(actions=None, answer="ok")

    tokens, token_masks, ar_masks, loss_masks, *_ = tokenizer.tokenize(prompt, None, None, answer="ok")

    assert token_masks.all()
    assert loss_masks.any()
    assert int(loss_masks.sum()) == len(postfix_tokens)
    assert tokens[-len(postfix_tokens) :].tolist() == postfix_tokens
    assert ar_masks[-len(postfix_tokens) :].all()


def test_fast_tokenizer_truncation_preserves_action_postfix():
    tokenizer = _make_fast_tokenizer(max_len=64)
    prompt = "x" * 500  # Force prefix overflow
    state = np.zeros((64,), dtype=np.float32)  # Long state string contributes to overflow
    actions = np.zeros((3, 2), dtype=np.float32)
    postfix_tokens = tokenizer._tokenize_postfix(actions=actions, answer=None)

    tokens, token_masks, ar_masks, loss_masks, *_ = tokenizer.tokenize(prompt, state, actions)

    assert token_masks.all()
    assert loss_masks.any()
    assert int(loss_masks.sum()) == len(postfix_tokens)
    assert tokens[-len(postfix_tokens) :].tolist() == postfix_tokens
    assert ar_masks[-len(postfix_tokens) :].all()


def test_fast_tokenizer_truncation_aligns_flow_prefix_with_supervised_prefix():
    tokenizer = _make_fast_tokenizer(max_len=64)
    prompt = "x" * 500  # Force prefix overflow
    state = np.zeros((64,), dtype=np.float32)
    actions = np.zeros((3, 2), dtype=np.float32)

    tokens, token_masks, _, loss_masks, flow_tokens, flow_masks = tokenizer.tokenize(prompt, state, actions)

    kept_prefix = tokens[token_masks & ~loss_masks]
    flow_prefix = flow_tokens[flow_masks]

    assert flow_prefix.tolist() == kept_prefix.tolist()
