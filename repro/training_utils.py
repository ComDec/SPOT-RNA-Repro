import os
import random
import string
import tempfile
import zipfile
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset


MODEL_PRESETS = {
    "paper-small": {
        "channels": 32,
        "num_blocks": 8,
        "fc_layers": 1,
        "fc_channels": 128,
        "bilstm_hidden": 0,
        "dilation_cycle": 0,
    },
    "model0": {
        "channels": 48,
        "num_blocks": 16,
        "fc_layers": 2,
        "fc_channels": 512,
        "bilstm_hidden": 0,
        "dilation_cycle": 0,
    },
    "model1": {
        "channels": 64,
        "num_blocks": 20,
        "fc_layers": 1,
        "fc_channels": 512,
        "bilstm_hidden": 0,
        "dilation_cycle": 0,
    },
    "model2": {
        "channels": 64,
        "num_blocks": 30,
        "fc_layers": 1,
        "fc_channels": 512,
        "bilstm_hidden": 0,
        "dilation_cycle": 0,
    },
    "model3": {
        "channels": 64,
        "num_blocks": 30,
        "fc_layers": 0,
        "fc_channels": 0,
        "bilstm_hidden": 200,
        "dilation_cycle": 0,
    },
    "model4": {
        "channels": 64,
        "num_blocks": 30,
        "fc_layers": 1,
        "fc_channels": 512,
        "bilstm_hidden": 0,
        "dilation_cycle": 5,
    },
}


class LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_sequence(sequence):
    return sequence.replace(" ", "").replace("\n", "").upper().replace("T", "U")


def one_hot_matrix(sequence):
    bases = np.array(list("AUCG"))
    matrix = []
    for base in sequence:
        if base in "AUCG":
            matrix.append((bases == base).astype(np.float32))
        else:
            matrix.append(np.array([-1, -1, -1, -1], dtype=np.float32))
    return np.array(matrix, dtype=np.float32)


def build_pair_feature(sequence):
    one_hot = one_hot_matrix(sequence)
    tiled = np.tile(one_hot[None, :, :], (one_hot.shape[0], 1, 1))
    feature = np.concatenate([tiled, np.transpose(tiled, (1, 0, 2))], axis=2)
    return np.transpose(feature.astype(np.float32), (2, 0, 1))


def standardize_feature_tensor(feature_tensor, feature_mean, feature_std):
    if feature_mean is None or feature_std is None:
        return feature_tensor
    return (feature_tensor - feature_mean[:, None, None]) / feature_std[:, None, None]


def get_dataset_cache_dir(archive_path):
    cache_dir = os.path.join(os.path.dirname(archive_path), ".spotrna_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_file_path(archive_path, dataset_type, split, suffix):
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    return os.path.join(
        get_dataset_cache_dir(archive_path),
        "{0}_{1}_{2}_{3}".format(archive_name, dataset_type, split, suffix),
    )


def atomic_torch_save(payload, path):
    fd, temp_path = tempfile.mkstemp(
        prefix="spotrna_", suffix=".tmp", dir=os.path.dirname(path)
    )
    os.close(fd)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def valid_pair_mask(sequence):
    length = len(sequence)
    mask = np.ones((length, length), dtype=np.float32)
    invalid_positions = [idx for idx, base in enumerate(sequence) if base not in "AUCG"]
    if invalid_positions:
        mask[invalid_positions, :] = 0
        mask[:, invalid_positions] = 0
    return np.triu(mask, 2)


def allowed_pair_mask(sequence):
    include_pairs = {
        "AU",
        "UA",
        "GC",
        "CG",
        "GU",
        "UG",
        "CC",
        "GG",
        "AG",
        "CA",
        "AC",
        "UU",
        "AA",
        "CU",
        "GA",
        "UC",
    }
    mask = np.zeros((len(sequence), len(sequence)), dtype=np.float32)
    for row_idx, left in enumerate(sequence):
        for col_idx, right in enumerate(sequence):
            if left + right in include_pairs:
                mask[row_idx, col_idx] = 1.0
    return mask


def parse_dot_bracket(structure):
    openers = "([{<" + string.ascii_uppercase
    closers = ")]}>" + string.ascii_lowercase
    close_to_open = {
        close_char: open_char for open_char, close_char in zip(openers, closers)
    }
    stacks = {open_char: [] for open_char in openers}
    pairs = []

    for idx, char in enumerate(structure):
        if char in stacks:
            stacks[char].append(idx)
        elif char in close_to_open:
            open_char = close_to_open[char]
            if stacks[open_char]:
                left = stacks[open_char].pop()
                right = idx
                pairs.append((min(left, right), max(left, right)))
    return sorted(set(pairs))


def parse_bpRNA_st(st_text):
    sequence = None
    structure = None
    sample_id = None
    for raw_line in st_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#Name:"):
            sample_id = line.split(":", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        if sequence is None:
            sequence = normalize_sequence(line)
        elif structure is None:
            structure = line.strip()
            break

    if sample_id is None or sequence is None or structure is None:
        raise ValueError("Unable to parse bpRNA .st sample")

    return sample_id, sequence, parse_dot_bracket(structure)


def parse_pdb_sequence(sequence_text):
    lines = [line.strip() for line in sequence_text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError("Unable to parse PDB sequence file")
    sample_id = lines[0][1:]
    sequence = normalize_sequence("".join(lines[1:]))
    return sample_id, sequence


def parse_pdb_labels(label_text):
    pairs = []
    for raw_line in label_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower().startswith("i"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        left = int(parts[0]) - 1
        right = int(parts[1]) - 1
        if left == right:
            continue
        pairs.append((min(left, right), max(left, right)))
    return sorted(set(pairs))


def build_label_matrix(length, pairs):
    matrix = np.zeros((length, length), dtype=np.float32)
    for left, right in pairs:
        if 0 <= left < right < length:
            matrix[left, right] = 1.0
    return matrix


def filter_pairs_by_sequence(pairs, sequence, min_separation=2):
    filtered = []
    valid_bases = set("AUCG")
    for left, right in pairs:
        if right - left < min_separation:
            continue
        if sequence[left] not in valid_bases or sequence[right] not in valid_bases:
            continue
        filtered.append((left, right))
    return sorted(set(filtered))


def remove_multiplet_pairs(pairs):
    grouped_pairs = multiplets_pairs(pairs)
    if not grouped_pairs:
        return sorted(set(pairs))
    pairs_to_remove = {pair for group in grouped_pairs for pair in group}
    return sorted(set(pair for pair in pairs if pair not in pairs_to_remove))


def flatten_pairs(pairs):
    flattened = []
    for pair in pairs:
        flattened.extend(pair)
    return flattened


def multiplets_pairs(pred_pairs):
    counts = Counter(flatten_pairs(pred_pairs))
    duplicates = {idx for idx, count in counts.items() if count > 1}
    grouped = defaultdict(list)
    for pair in pred_pairs:
        left, right = pair
        if left in duplicates:
            grouped[left].append(pair)
        if right in duplicates and right != left:
            grouped[right].append(pair)
    return list(grouped.values())


def multiplets_free_bp(pred_pairs, scores):
    original_count = len(pred_pairs)
    removed_pairs = []
    grouped_pairs = multiplets_pairs(pred_pairs)
    while grouped_pairs:
        to_remove = []
        for group in grouped_pairs:
            group_scores = [scores[left, right] for left, right in group]
            worst_pair = group[group_scores.index(min(group_scores))]
            to_remove.append(worst_pair)
            removed_pairs.append(worst_pair)
        pred_pairs = [pair for pair in pred_pairs if pair not in to_remove]
        grouped_pairs = multiplets_pairs(pred_pairs)

    if original_count != len(pred_pairs) + len(set(removed_pairs)):
        raise AssertionError("Multiplet filtering count mismatch")
    return pred_pairs, removed_pairs


def logits_to_pairs(logits, sequence, threshold):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    probabilities = np.multiply(probabilities, allowed_pair_mask(sequence))
    probabilities = np.multiply(probabilities, valid_pair_mask(sequence))
    tri_indices = np.triu_indices(probabilities.shape[0], k=2)
    pred_pairs = []
    for row_idx, col_idx in zip(tri_indices[0], tri_indices[1]):
        if probabilities[row_idx, col_idx] >= threshold:
            pred_pairs.append((row_idx, col_idx))
    pred_pairs, _ = multiplets_free_bp(pred_pairs, probabilities)
    return set(pred_pairs)


def confusion_from_pairs(true_pairs, pred_pairs, valid_positions):
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    tn = int(valid_positions - tp - fp - fn)
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        (2 * precision * sensitivity / (precision + sensitivity))
        if (precision + sensitivity)
        else 0.0
    )
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        mcc = 0.0
    else:
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denominator)
    return {
        "precision": precision,
        "sensitivity": sensitivity,
        "f1": f1_score,
        "mcc": mcc,
    }


def write_ct_file(pairs, sequence, sample_id, output_dir):
    col1 = np.arange(1, len(sequence) + 1, 1)
    col2 = np.array(list(sequence))
    col3 = np.arange(0, len(sequence), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(sequence), dtype=int)
    for left, right in pairs:
        col5[left] = int(right) + 1
        col5[right] = int(left) + 1
    col6 = np.arange(1, len(sequence) + 1, 1)
    rows = np.vstack(
        (
            np.char.mod("%d", col1),
            col2,
            np.char.mod("%d", col3),
            np.char.mod("%d", col4),
            np.char.mod("%d", col5),
            np.char.mod("%d", col6),
        )
    ).T
    np.savetxt(
        os.path.join(output_dir, sample_id + ".ct"),
        rows,
        delimiter="\t\t",
        fmt="%s",
        header=str(len(sequence)) + "\t\t" + sample_id + "\t\tSPOT-RNA output\n",
        comments="",
    )


def write_bpseq_file(pairs, sequence, sample_id, output_dir):
    col1 = np.arange(1, len(sequence) + 1, 1)
    col2 = np.array(list(sequence))
    col3 = np.zeros(len(sequence), dtype=int)
    for left, right in pairs:
        col3[left] = int(right) + 1
        col3[right] = int(left) + 1
    rows = np.vstack((np.char.mod("%d", col1), col2, np.char.mod("%d", col3))).T
    np.savetxt(
        os.path.join(output_dir, sample_id + ".bpseq"),
        rows,
        delimiter=" ",
        fmt="%s",
        header="#" + sample_id,
        comments="",
    )


def read_fasta_records(path):
    records = []
    current_id = None
    current_seq = []
    with open(path) as fasta_file:
        for raw_line in fasta_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(
                        (current_id, normalize_sequence("".join(current_seq)))
                    )
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        records.append((current_id, normalize_sequence("".join(current_seq))))
    if not records:
        raise ValueError("No FASTA records found in {0}".format(path))
    return records


class RNAPairDataset(Dataset):
    def __init__(
        self,
        archive_path,
        dataset_type,
        split,
        limit=0,
        drop_multiplets=False,
        feature_mean=None,
        feature_std=None,
    ):
        self.archive_path = archive_path
        self.dataset_type = dataset_type
        self.split = split
        self.drop_multiplets = drop_multiplets
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.entries = self._load_or_build_entries()
        if limit:
            self.entries = self.entries[:limit]

    def _build_entries(self):
        with zipfile.ZipFile(self.archive_path) as archive:
            names = archive.namelist()
            if self.dataset_type == "pdb":
                seq_prefix = "PDB_dataset/{0}_sequences/".format(self.split)
                label_prefix = "PDB_dataset/{0}_labels/".format(self.split)
                sequences = {}
                labels = {}
                for name in names:
                    if name.endswith("/"):
                        continue
                    if name.startswith(seq_prefix):
                        sequences[os.path.basename(name)] = name
                    elif name.startswith(label_prefix):
                        labels[os.path.basename(name)] = name
                common_ids = sorted(set(sequences) & set(labels))
                return [
                    (sample_id, sequences[sample_id], labels[sample_id])
                    for sample_id in common_ids
                ]

            if self.dataset_type == "bpRNA":
                prefix = "bpRNA_dataset/{0}/".format(self.split)
                members = [
                    name
                    for name in names
                    if name.startswith(prefix) and name.endswith(".st")
                ]
                return [
                    (os.path.basename(name).rsplit(".", 1)[0], name, None)
                    for name in sorted(members)
                ]

        raise ValueError("Unsupported dataset type: {0}".format(self.dataset_type))

    def _load_or_build_entries(self):
        cache_path = cache_file_path(
            self.archive_path, self.dataset_type, self.split, "parsed.pt"
        )
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        indexed_entries = self._build_entries()
        parsed_entries = []
        with zipfile.ZipFile(self.archive_path) as archive:
            for sample_id, sequence_member, label_member in indexed_entries:
                if self.dataset_type == "pdb":
                    parsed_id, sequence = parse_pdb_sequence(
                        archive.read(sequence_member).decode("utf-8")
                    )
                    pairs = parse_pdb_labels(archive.read(label_member).decode("utf-8"))
                else:
                    parsed_id, sequence, pairs = parse_bpRNA_st(
                        archive.read(sequence_member).decode("utf-8")
                    )
                parsed_entries.append(
                    {
                        "id": parsed_id,
                        "sequence": sequence,
                        "pairs": pairs,
                    }
                )
        atomic_torch_save(parsed_entries, cache_path)
        return parsed_entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        parsed_id = entry["id"]
        sequence = entry["sequence"]
        pairs = entry["pairs"]

        pairs = filter_pairs_by_sequence(pairs, sequence)
        if self.drop_multiplets:
            pairs = remove_multiplet_pairs(pairs)

        features = build_pair_feature(sequence)
        features = standardize_feature_tensor(
            features, self.feature_mean, self.feature_std
        )
        targets = build_label_matrix(len(sequence), pairs)[None, :, :]
        mask = valid_pair_mask(sequence)[None, :, :]

        return {
            "id": parsed_id,
            "sequence": sequence,
            "pairs": pairs,
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "length": len(sequence),
        }


def collate_rna_batch(batch):
    max_len = max(item["length"] for item in batch)
    batch_size = len(batch)
    features = torch.zeros(batch_size, 8, max_len, max_len, dtype=torch.float32)
    targets = torch.zeros(batch_size, 1, max_len, max_len, dtype=torch.float32)
    masks = torch.zeros(batch_size, 1, max_len, max_len, dtype=torch.float32)

    ids = []
    sequences = []
    pairs = []
    lengths = []

    for row_idx, item in enumerate(batch):
        length = item["length"]
        features[row_idx, :, :length, :length] = item["features"]
        targets[row_idx, :, :length, :length] = item["targets"]
        masks[row_idx, :, :length, :length] = item["mask"]
        ids.append(item["id"])
        sequences.append(item["sequence"])
        pairs.append(set(item["pairs"]))
        lengths.append(length)

    return {
        "ids": ids,
        "sequences": sequences,
        "pairs": pairs,
        "lengths": lengths,
        "features": features,
        "targets": targets,
        "mask": masks,
    }


def compute_feature_stats(dataset):
    if not len(dataset):
        return np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)

    cache_path = cache_file_path(
        dataset.archive_path, dataset.dataset_type, dataset.split, "feature_stats.pt"
    )
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        return cached["mean"], cached["std"]

    feature_sum = np.zeros(8, dtype=np.float64)
    feature_sq_sum = np.zeros(8, dtype=np.float64)
    feature_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = sample["features"].numpy()
        feature_sum += features.sum(axis=(1, 2))
        feature_sq_sum += np.square(features).sum(axis=(1, 2))
        feature_count += features.shape[1] * features.shape[2]

    mean = feature_sum / max(feature_count, 1)
    variance = (feature_sq_sum / max(feature_count, 1)) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))
    payload = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    atomic_torch_save(payload, cache_path)
    return payload["mean"], payload["std"]


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.25):
        super().__init__()
        self.norm1 = LayerNorm2D(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = LayerNorm2D(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class AxialBiLSTM2D(nn.Module):
    def __init__(self, channels, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.row_lstm = nn.LSTM(
            channels, hidden_size, batch_first=True, bidirectional=True
        )
        self.col_lstm = nn.LSTM(
            channels, hidden_size, batch_first=True, bidirectional=True
        )
        self.proj = nn.Conv2d(hidden_size * 4, channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        row_input = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size * height, width, channels)
        )
        row_output, _ = self.row_lstm(row_input)
        row_output = row_output.view(
            batch_size, height, width, self.hidden_size * 2
        ).permute(0, 3, 1, 2)

        col_input = (
            x.permute(0, 3, 2, 1)
            .contiguous()
            .view(batch_size * width, height, channels)
        )
        col_output, _ = self.col_lstm(col_input)
        col_output = col_output.view(
            batch_size, width, height, self.hidden_size * 2
        ).permute(0, 3, 2, 1)

        merged = torch.cat([row_output, col_output], dim=1)
        return self.proj(merged)


class PointwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.norm = LayerNorm2D(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        return self.conv(x)


class PaperInspiredSPOTRNA(nn.Module):
    def __init__(
        self,
        channels=32,
        num_blocks=8,
        fc_layers=1,
        fc_channels=128,
        bilstm_hidden=0,
        dilation_cycle=0,
    ):
        super().__init__()
        self.stem = nn.Conv2d(8, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for block_idx in range(num_blocks):
            dilation = 1
            if dilation_cycle:
                dilation = 2 ** (block_idx % dilation_cycle)
            self.blocks.append(ResidualBlock(channels, dilation=dilation))
        self.post_norm = LayerNorm2D(channels)
        self.axial_bilstm = (
            AxialBiLSTM2D(channels, bilstm_hidden) if bilstm_hidden else None
        )

        pointwise_layers = []
        in_channels = channels
        for _ in range(fc_layers):
            pointwise_layers.append(PointwiseBlock(in_channels, fc_channels))
            in_channels = fc_channels
        self.pointwise_layers = nn.ModuleList(pointwise_layers)
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.elu(self.post_norm(x))
        if self.axial_bilstm is not None:
            x = self.axial_bilstm(x)
        for layer in self.pointwise_layers:
            x = layer(x)
        logits = self.head(x)
        return 0.5 * (logits + logits.transpose(-1, -2))


def masked_bce_loss(logits, targets, mask, positive_weight=1.0):
    if positive_weight == "auto":
        positive_count = (targets * mask).sum()
        negative_count = ((1.0 - targets) * mask).sum()
        pos_weight_value = (negative_count / positive_count.clamp_min(1.0)).detach()
    else:
        pos_weight_value = torch.tensor(float(positive_weight), device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_value,
    )
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def probabilities_to_pairs(probabilities, threshold):
    candidate_matrix = np.triu(probabilities >= threshold, k=2)
    row_indices, col_indices = np.where(candidate_matrix)
    pred_pairs = list(zip(row_indices.tolist(), col_indices.tolist()))
    pred_pairs, _ = multiplets_free_bp(pred_pairs, probabilities)
    return set(pred_pairs)


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    samples = []
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(features)
        batch_loss = masked_bce_loss(logits, targets, mask)
        total_loss += batch_loss.item() * features.size(0)
        total_count += features.size(0)

        for idx, length in enumerate(batch["lengths"]):
            sequence = batch["sequences"][idx]
            probabilities = (
                torch.sigmoid(logits[idx, 0, :length, :length]).detach().cpu().numpy()
            )
            probabilities = (
                probabilities * allowed_pair_mask(sequence) * valid_pair_mask(sequence)
            )
            samples.append(
                {
                    "sequence": sequence,
                    "true_pairs": batch["pairs"][idx],
                    "valid_positions": int(
                        batch["mask"][idx, 0, :length, :length].sum().item()
                    ),
                    "probabilities": probabilities,
                }
            )

    return samples, total_loss / max(total_count, 1)


def compute_metrics_for_threshold(samples, threshold):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    for sample in samples:
        pred_pairs = probabilities_to_pairs(sample["probabilities"], threshold)
        tp, fp, fn, tn = confusion_from_pairs(
            sample["true_pairs"], pred_pairs, sample["valid_positions"]
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    metrics["threshold"] = threshold
    return metrics


def search_best_threshold(
    model,
    dataloader,
    device,
    metric_name="f1",
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)
    samples, loss = collect_predictions(model, dataloader, device)
    best_metrics = None
    for threshold in thresholds:
        metrics = compute_metrics_for_threshold(samples, float(threshold))
        metrics["loss"] = loss
        if best_metrics is None:
            best_metrics = metrics
            continue
        if metrics[metric_name] > best_metrics[metric_name]:
            best_metrics = metrics
            continue
        if (
            metrics[metric_name] == best_metrics[metric_name]
            and metrics["mcc"] > best_metrics["mcc"]
        ):
            best_metrics = metrics
    return best_metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold):
    samples, loss = collect_predictions(model, dataloader, device)
    metrics = compute_metrics_for_threshold(samples, threshold)
    metrics["loss"] = loss
    return metrics
