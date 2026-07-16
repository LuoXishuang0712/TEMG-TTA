# TEMG-TTA

This is the official implementation of TEMG-TTA, a method for blockchain anomaly detection based on temporal motif matching and test-time adaptation.

>
> Related paper: **Temporal Motif-aware Graph Test-time Adaptation for OOD Blockchain Anomaly Detection**
>
> *Accepted to IJCAI-ECAI 2026, Special Track on AI for Social Good.*
>

## Folder Structure

1. GADBenchTTA
    * Where we implementing our TTA methods, please config environment according to [GADBench_Readme](GADBenchTTA/readme.md).
    * There are several commands in [script](GADBenchTTA/run_batch.sh) as examples.
2. motifs_matching
    * The code used to match motifs.
    * See [Next Section](#motifs-matching) for more details.

## Data Available

The pre-built DGL graph and motifs matrix can be found at:

Google Drive:
https://drive.google.com/drive/folders/1G5de5Y5aWAZBTdBpS3HO_dNtp2qcUrPM?usp=sharing

Baidu Netdisk:
https://pan.baidu.com/s/1MuAIOI-ubVR_KJFUoqc80Q?pwd=v5rw

## Motifs Matching

### Environment Setup

0. Please install the Rust runtime first. You can use [rustup](https://rustup.rs/) to do so.
1. Run `cargo build --release`
2. Run `target/release/rust_motifs --help` for further information.

### Input / Output

* Input example

```
<from_node_id> <to_node_id> <timestamp>
...
```

* Output example

```
<node_1> <node_2> <node_3> <motif_id> [<timestamp>]!...
```

## Citation

```bibtex
@misc{he2026temporalmotifawaregraphtesttime,
      title={Temporal Motif-aware Graph Test-time Adaptation for OOD Blockchain Anomaly Detection}, 
      author={Runang He and Tongya Zheng and Huiling Peng and Yuanyu Wan and Bingde Hu and Jiawei Chen and Canghong Jin and Mingli Song and Can Wang},
      year={2026},
      eprint={2605.29526},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.29526}, 
}
```
