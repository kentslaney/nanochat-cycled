"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import os
import json
import random
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into the local cache directory, and
    read them with pyarrow. Under torchrun, only one rank downloads, the others wait.
    """
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    # the manifest is written last, so its existence means the download completed
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single rank acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}"
                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class RenderCache:
    """
    Precomputes tokenizer.render_conversation(ids, mask, positions) for every row of
    `dataset` and memory-maps the result from disk, so training never re-tokenizes or
    re-runs the (pure-Python, per-token) cycled-RoPE position loop on the fly. Built once
    per unique `cache_key`, guarded by a FileLock exactly like load_hub_dataset above:
    under torchrun, only one rank renders and the others block, then all reread.
    """

    def __init__(self, dataset, tokenizer, message_stride, cache_key):
        base_dir = get_base_dir()
        cache_dir = os.path.join(base_dir, "sft_render_cache", cache_key)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            os.makedirs(cache_dir, exist_ok=True)
            with FileLock(manifest_path + ".lock"):
                # only a single rank acquires the lock and renders, the others block
                # here and then skip rendering because they recheck the manifest
                if not os.path.exists(manifest_path):
                    self._build(dataset, tokenizer, message_stride, cache_dir, manifest_path)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.message_stride = manifest["message_stride"]
        self.num_conversations = manifest["num_conversations"]
        self.offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        self.ids = np.load(os.path.join(cache_dir, "ids.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(cache_dir, "mask.npy"), mmap_mode="r")
        self.positions = None
        if self.message_stride > 0:
            self.positions = np.load(os.path.join(cache_dir, "positions.npy"), mmap_mode="r")

    @staticmethod
    def _build(dataset, tokenizer, message_stride, cache_dir, manifest_path):
        n = len(dataset)
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        ids_parts, mask_parts, pos_parts = [], [], []
        for i in range(n):
            if message_stride > 0:
                ids, mask, positions = tokenizer.render_conversation(
                    dataset[i], message_stride=message_stride, return_positions=True)
            else:
                ids, mask = tokenizer.render_conversation(dataset[i])
                positions = None
            ids_parts.append(np.asarray(ids, dtype=np.int32))
            mask_parts.append(np.asarray(mask, dtype=np.int8))
            if positions is not None:
                pos_parts.append(np.asarray(positions, dtype=np.int32))
            offsets[i + 1] = offsets[i] + len(ids)
            if i % 20000 == 0:
                print(f"Rendering SFT cache ({cache_dir}): {i}/{n} conversations")
        empty = lambda dtype: np.zeros(0, dtype=dtype)
        np.save(os.path.join(cache_dir, "offsets.npy"), offsets)
        np.save(os.path.join(cache_dir, "ids.npy"), np.concatenate(ids_parts) if ids_parts else empty(np.int32))
        np.save(os.path.join(cache_dir, "mask.npy"), np.concatenate(mask_parts) if mask_parts else empty(np.int8))
        if message_stride > 0:
            np.save(os.path.join(cache_dir, "positions.npy"), np.concatenate(pos_parts) if pos_parts else empty(np.int32))
        with open(manifest_path, "w") as f:
            json.dump({"num_conversations": n, "message_stride": message_stride}, f)

    def __len__(self):
        return self.num_conversations

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        ids = self.ids[start:end].tolist()
        mask = self.mask[start:end].tolist()
        positions = self.positions[start:end].tolist() if self.positions is not None else None
        return ids, mask, positions


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="all", split="auxiliary_train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="all", split="auxiliary_train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
