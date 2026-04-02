import argparse
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.FastaMLtoSL import FastaMLtoSL
from utils.utils import create_tfr_files, prob_to_secondary_structure


MODEL_URLS = [
    "https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz",
    "https://app.nihaocloud.com/f/fbf3315a91d542c0bdc2/?dl=1",
]
NUM_MODELS = 5
DEFAULT_CPU_THREADS = max(1, min(16, os.cpu_count() or 1))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run SPOT-RNA secondary structure prediction on one or more FASTA sequences.",
    )
    parser.add_argument(
        "--input",
        "--inputs",
        dest="inputs",
        default="sample_inputs/single_seq.fasta",
        type=str,
        help="Path to input FASTA file. Multiline FASTA is supported.",
        metavar="",
    )
    parser.add_argument(
        "--output",
        "--outputs",
        dest="outputs",
        default="outputs",
        type=str,
        help="Directory where SPOT-RNA writes .ct, .bpseq, and .prob outputs.",
        metavar="",
    )
    parser.add_argument(
        "--gpu",
        default=-1,
        type=int,
        help="GPU index to use. Keep -1 to run on CPU.",
        metavar="",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate 2D structure plots with VARNA if Java is available.",
    )
    parser.add_argument(
        "--motifs",
        action="store_true",
        help="Generate motif files with bpRNA if Perl Graph.pm is available.",
    )
    parser.add_argument(
        "--cpu",
        default=DEFAULT_CPU_THREADS,
        type=int,
        help="Number of CPU threads to use in CPU mode. Default uses up to 16 threads.",
        metavar="",
    )
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=np.float128)))


def ensure_models(base_path):
    model_dir = os.path.join(base_path, "SPOT-RNA-models")
    expected_files = []
    for model_idx in range(NUM_MODELS):
        expected_files.extend(
            [
                os.path.join(model_dir, "model{0}.meta".format(model_idx)),
                os.path.join(model_dir, "model{0}.index".format(model_idx)),
                os.path.join(
                    model_dir, "model{0}.data-00000-of-00001".format(model_idx)
                ),
            ]
        )

    if all(os.path.exists(path) for path in expected_files):
        return model_dir

    archive_path = os.path.join(base_path, "SPOT-RNA-models.tar.gz")
    last_error = None
    print("\nDownloading pretrained SPOT-RNA models...")
    for url in MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, archive_path)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            "Unable to download pretrained models: {0}".format(last_error)
        )

    with tarfile.open(archive_path, "r:gz") as model_archive:
        model_archive.extractall(base_path)
    os.remove(archive_path)

    if not all(os.path.exists(path) for path in expected_files):
        raise RuntimeError(
            "Model download finished, but some checkpoint files are still missing."
        )

    return model_dir


def prepare_input_fasta(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input FASTA file not found: {0}".format(input_path))

    temp_dir = tempfile.mkdtemp(prefix="spotrna_")
    temp_input = os.path.join(temp_dir, os.path.basename(input_path))
    shutil.copyfile(input_path, temp_input)
    FastaMLtoSL(temp_input)
    return temp_dir, temp_input


def load_sequences(input_path):
    with open(input_path) as file_obj:
        input_data = [
            line.strip() for line in file_obj.read().splitlines() if line.strip()
        ]

    if not input_data or len(input_data) % 2 != 0:
        raise ValueError("Input FASTA must contain complete header/sequence pairs.")

    count = int(len(input_data) / 2)
    ids = [input_data[2 * i].replace(">", "") for i in range(count)]
    sequences = {}
    for i, seq_id in enumerate(ids):
        sequences[seq_id] = (
            input_data[2 * i + 1].replace(" ", "").upper().replace("T", "U")
        )

    return count, sequences


def main():
    start = time.time()
    args = parse_args()
    base_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = ensure_models(base_path)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == -1 else str(args.gpu)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    output_path = args.outputs
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_path, output_path)
    os.makedirs(output_path, exist_ok=True)
    args.outputs = output_path

    temp_dir = None
    tfrecord_path = None
    try:
        temp_dir, prepared_input = prepare_input_fasta(args.inputs)
        input_file = os.path.basename(prepared_input)

        create_tfr_files(prepared_input, base_path, input_file)
        tfrecord_path = os.path.join(
            base_path, "input_tfr_files", input_file + ".tfrecords"
        )

        count, sequences = load_sequences(prepared_input)
        test_loc = [tfrecord_path]
        outputs = {}
        mask = {}

        for model_idx in range(NUM_MODELS):
            if args.gpu == -1:
                config = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=args.cpu,
                    inter_op_parallelism_threads=args.cpu,
                )
            else:
                config = tf.compat.v1.ConfigProto()
                config.allow_soft_placement = True
                config.log_device_placement = False
                config.gpu_options.allow_growth = True

            print("\nPredicting for SPOT-RNA model " + str(model_idx))
            with tf.compat.v1.Session(config=config) as sess:
                saver = tf.compat.v1.train.import_meta_graph(
                    os.path.join(model_dir, "model" + str(model_idx) + ".meta")
                )
                saver.restore(sess, os.path.join(model_dir, "model" + str(model_idx)))
                graph = tf.compat.v1.get_default_graph()
                init_test = graph.get_operation_by_name("make_initializer_2")
                tmp_out = graph.get_tensor_by_name(
                    "output_FC/fully_connected/BiasAdd:0"
                )
                name_tensor = graph.get_tensor_by_name("tensors_2/component_0:0")
                rna_name = graph.get_tensor_by_name("IteratorGetNext:0")
                label_mask = graph.get_tensor_by_name("IteratorGetNext:4")
                sess.run(init_test, feed_dict={name_tensor: test_loc})

                pbar = tqdm(total=count)
                while True:
                    try:
                        out = sess.run(
                            [tmp_out, rna_name, label_mask], feed_dict={"dropout:0": 1}
                        )
                        out[1] = out[1].decode()
                        mask[out[1]] = out[2]
                        if model_idx == 0:
                            outputs[out[1]] = [sigmoid(out[0])]
                        else:
                            outputs[out[1]].append(sigmoid(out[0]))
                        pbar.update(1)
                    except tf.errors.OutOfRangeError:
                        break
                pbar.close()
            tf.compat.v1.reset_default_graph()

        print("\nPost Processing and Saving Output")
        for rna_id in list(outputs.keys()):
            ensemble_output = np.mean(outputs[rna_id], 0)
            prob_to_secondary_structure(
                ensemble_output,
                mask[rna_id],
                sequences[rna_id],
                rna_id,
                args,
                base_path,
            )

        print("\nFinished!")
        print("\nProcesssing Time {} seconds".format(time.time() - start))
    finally:
        if tfrecord_path and os.path.exists(tfrecord_path):
            os.remove(tfrecord_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
