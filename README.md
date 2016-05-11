# Before all
This is a combination of torch-rnn-server and arxiv-sanity-preserver. Here I use the arxiv-sanity code to download the pdf and process the pdfs to text files. And follow the torch-rnn-server pipeline to train the model. To change the paper you want to download, change the query in `fetch_papers.py` 

# torch-rnn-server

This is a small server that works with the Atom package [`rnn-writer`](https://github.com/robinsloan/rnn-writer) to provide responsive, inline "autocomplete" powered by a recurrent neural network trained on a corpus of sci-fi stories, or another corpus of your choosing.

More accurately: it's a set of shims laid beneath Justin Johnson's indispensable `torch-rnn` package.

I explain what this project is all about [here](https://www.robinsloan.com/note/writing-with-the-machine); it's probably worth reading that before continuing.

###Installation

There are a couple of different ways to get `torch-rnn-server` running, but no matter what, you'll need to install Torch, the scientific computing framework that powers the whole operation. Those instructions are below, in the original `torch-rnn` README.

After completing those steps, you'll need to install Ben Glard's [`waffle`](https://github.com/benglard/waffle) project to power the web server:

```
luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec
```

###Training and models

After installing Torch and all of `torch-rnn`'s dependencies, you can train a model on a corpus of your choosing; those instructions are below, in the original `torch-rnn` README. **Alternatively, you can download a pre-trained model** derived from ~150MB of old sci-fi stories:

```
cd checkpoints
wget http://from.robinsloan.com/rnn-writer/scifi-model.zip
unzip scifi-model.zip
```

(You can read a bit more about the corpus and find a link to download it [here](https://www.robinsloan.com/note/writing-with-the-machine).)

###Running the server

Finally! You can start the server with

```
th server.lua
```

and try

```
curl "http://0.0.0.0/generate?start_text=It%20was%20a%20dark&n=3"
```

If you see a JSON response offering strange sentences, it means everything is working, and it's onward to [`rnn-writer`](https://github.com/robinsloan/rnn-writer)!

Standard `torch-rnn` README continues below.

# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn).

You can find documentation for the RNN and LSTM modules [here](doc/modules.md); they have no dependencies other than `torch`
and `nn`, so they should be easy to integrate into existing projects.

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see
the [Benchmark](#benchmarks) section below.


# Installation

## System setup

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

You'll need to install the header files for Python 2.7 and the HDF5 library. On Ubuntu you should be able to install
like this:

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

## Python setup

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

The preprocessing script is written in Python 2.7; its dependencies are in the file `requirements.txt`.
You can install these dependencies in a virtual environment like this:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

## Lua setup

**`torch-rnn-server note`: You can't skip this :(**

The main modeling code is written in Lua using [torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)

After installing torch, you can install / update these packages by running the following:

```bash
# Install most things using luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson

# We need to install torch-hdf5 from GitHub

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
```

### CUDA support (Optional)

**`torch-rnn-server note`: If you skip this, everything will be slowww :(**

To enable GPU acceleration with CUDA, you'll need to install CUDA 6.5 or higher and the following Lua packages:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update them by running:

```bash
luarocks install cutorch
luarocks install cunn
```

## OpenCL support (Optional)
To enable GPU acceleration with OpenCL, you'll need to install the following Lua packages:
- [cltorch](https://github.com/hughperkins/cltorch)
- [clnn](https://github.com/hughperkins/clnn)

You can install / update them by running:

```bash
luarocks install cltorch
luarocks install clnn
```

## OSX Installation
Jeff Thompson has written a very detailed installation guide for OSX that you [can find here](http://www.jeffreythompson.org/blog/2016/03/25/torch-rnn-mac-install/).

**`torch-rnn-server note`: You can STOP HERE if you're using a pretrained model.**

# Usage
To train a model and use it to generate new text, you'll need to follow three simple steps:

## Step 0: Download data from arxiv and process the raw data.

- Run `fetch_papers.py` to query arxiv API and create a file db.p that contains all information for each paper. This script is where you would modify the query, indicating which parts of arxiv you'd like to use. Note that if you're trying to pull too many papers arxiv will start to rate limit you. You may have to run the script multiple times, and I recommend using the arg --start_index to restart where you left off when you were last interrupted by arxiv.
- Run `download_pdf.py`, which iterates over all papers in parsed pickle and downloads the papers into folder pdf
- Run `parse_pdf_to_text.py` to export all text from pdfs to files in txt 
- Run `combine_pdfs.py` to combine all the pdf texts to a single txt file.

## Step 1: Preprocess the data
You can use any text file for training models. Before training, you'll need to preprocess the data using the script
`scripts/preprocess.py`; this will generate an HDF5 file and JSON file containing a preprocessed version of the data.

If you have training data stored in `my_data.txt`, you can run the script like this:

```bash
python scripts/preprocess.py \
  --input_txt arxiv_data.txt \
  --output_h5 arxiv_data.h5 \
  --output_json arxiv_data.json
```

This will produce files `my_data.h5` and `my_data.json` that will be passed to the training script.

There are a few more flags you can use to configure preprocessing; [read about them here](doc/flags.md#preprocessing)

## Step 2: Train the model
After preprocessing the data, you'll need to train the model using the `train.lua` script. This will be the slowest step.
You can run the training script like this:

```bash
th train.lua -input_h5 arxiv_data.h5 -input_json arxiv_data.json
```

This will read the data stored in `arxiv_data.h5` and `arxiv_data.json`, run for a while, and save checkpoints to files with
names like `checkpoints/checkpoint_1000.t7`.

You can change the RNN model type, hidden state size, and number of RNN layers like this:

```bash
th train.lua -input_h5 arxiv_data.h5 -input_json arxiv_data.json -model_type rnn -num_layers 3 -rnn_size 256
```

By default this will run in GPU mode using CUDA; to run in CPU-only mode, add the flag `-gpu -1`.

To run with OpenCL, add the flag `-gpu_backend opencl`.

There are many more flags you can use to configure training; [read about them here](doc/flags.md#training).

## Step 3: Sample from the model
After training a model, you can generate new text by sampling from it using the script `sample.lua`. Run it like this:

```bash
th sample.lua -checkpoint checkpoints/checkpoint_10000.t7
```

This will load the trained checkpoint `cv/checkpoint_10000.t7` from the previous step, sample 2000 characters from it,
and print the results to the console.

By default the sampling script will run in GPU mode using CUDA; to run in CPU-only mode add the flag `-gpu -1` and
to run in OpenCL mode add the flag `-gpu_backend opencl`.

There are more flags you can use to configure sampling; [read about them here](doc/flags.md#sampling).

# Benchmarks
To benchmark `torch-rnn` against `char-rnn`, we use each to train LSTM language models for the tiny-shakespeare dataset
with 1, 2 or 3 layers and with an RNN size of 64, 128, 256, or 512. For each we use a minibatch size of 50, a sequence
length of 50, and no dropout. For each model size and for both implementations, we record the forward/backward times and
GPU memory usage over the first 100 training iterations, and use these measurements to compute the mean time and memory
usage.

All benchmarks were run on a machine with an Intel i7-4790k CPU, 32 GB main memory, and a Titan X GPU.

Below we show the forward/backward times for both implementations, as well as the mean speedup of `torch-rnn` over
`char-rnn`. We see that `torch-rnn` is faster than `char-rnn` at all model sizes, with smaller models giving a larger
speedup; for a single-layer LSTM with 128 hidden units, we achieve a **1.9x speedup**; for larger models we achieve about
a 1.4x speedup.

<img src='imgs/lstm_time_benchmark.png' width="800px">

Below we show the GPU memory usage for both implementations, as well as the mean memory saving of `torch-rnn` over
`char-rnn`. Again `torch-rnn` outperforms `char-rnn` at all model sizes, but here the savings become more significant for
larger models: for models with 512 hidden units, we use **7x less memory** than `char-rnn`.

<img src='imgs/lstm_memory_benchmark.png' width="800px">


# TODOs
- Get rid of Python / JSON / HDF5 dependencies?
