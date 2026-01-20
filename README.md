# TEMG-TTA

This is the official implementation of TEMG-TTA, a method for blockchain anomaly detection based on temporal motif matching and test-time adaptation.

>
> Related paper: (TBD)
>

## Folder Structure

1. GADBenchTTA
    * Where we implementing our TTA methods, please config environment according to [GADBench_Readme](GADBenchTTA/readme.md).
    * There are several commands in [script](GADBenchTTA/run_batch.sh) as examples.
2. motifs_matching
    * The code used to match motifs.
    * See [Next Section](#motifs-matching) for more details.

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
