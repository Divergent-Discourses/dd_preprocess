# Image Preprocessing Pipeline

Preprocesses images of written documents to prepare them for optical character recognition (OCR) or handwritten text recognition (HTR).

Aims to obtain more accurate transcriptions by making text more machine-readable.

Originally written for OCR/HTR of historical documents, predominantly Tibetan-language newspaper pages from 1950-60s.

## About

#### Step 1: Prepares images to meet Transkribus upload requirements

- Converts all images to JPEG format
- Ensures each image is maximum 300 DPI
- Increases size of images so that at least one dimension is 2500 pixels
- Ensures each image is maximum 10 MB

#### Step 2: Preprocesses images to be more accurately recognised by an optical character recognition (OCR) model

Pipeline originally designed to preprocess and improve readability of old scans/microfiche images of historical newspaper articles.

- Converts images to greyscale
- Applies fast non-local means denoising
- [optional: use -ce flag] Performs contrast stretching and the adaptive histogram equalisation (CLAHE) contrast enhancement method
- Performs Sauvola binarisation (local thresholding approach)
- Deskews images using projection profiling

## Installation

Using the command line, navigate to the location in which you wish to install the code. Then, download the code.

```bash
git clone https://github.com/Divergent-Discourses/dd_preprocess.git
```

Create a virtual environment.

```bash
conda create -n dd_preprocess python=3.10.13
```

Activate the environment.

```bash
conda activate dd_preprocess
```

Using the command line, navigate to the location of this repository. Then, install required packages.

```bash
cd dd_preprocess
pip install -r requirements.txt
```


## Usage

Place all images you want to preprocess in a directory. The directory can contain sub-directories if you want to keep image sub-groups.


Run the script from the command line once you have navigated to the location of this python file like this:

```bash
python dd_preprocess.py path/to/source/directory path/to/destination/directory
```

- **Source directory path:** The path to the folder which stores the images you want to preprocess

- **Target directory path:** The path to the folder which will store the preprocessed images - this doesn't have to exist yet. It just needs to include the desired path to/name for the folder


After you've typed that (before pressing enter), you can optionally include the following flags:

- **--k_val /  -k [float] :** Modify the K-value used during Sauvola binarisation (default: 0.14)

- **--window_size / -w [int] :** Modify the window size used during Sauvola binarisation. Should not be an even value (default: 21)

- **--contrast_enhance / -ce :** Use flag if you want to contrast stretch and enhance contrast of images within pipeline. Default is not to use this as it tends to introduce speckling.

- **--basic_only / -b :** Use flag if you only want to meet basic Transkribus upload requirements and do not want to do further preprocessing like binarisation (e.g. for evaluation
purposes). If you use this flag, the other optional flags are irrelevant as they apply to the further preprocessing pipeline.


For best results, you will need to tune **k_val** and **window_size** to values which work best for your materials. Default values were found to be the best compromise across varying image qualities (including red/coloured text, heavily stained/noisy pages and good quality images). For relatively clean images, try a k_val of **0.24** and window_size of **11**. 


Remember to deactivate the virtual environment once you're done.

```bash
conda deactivate dd_preprocess
```

#### For example...

You could use this command to adjust the k-value and window size used during binarisation to alter the quality of images outputted:

```bash
python dd_preprocess.py path/to/source/directory path/to/destination/directory --k_val 0.22 --window_size 301
```

You could use this command to only meet the basic Transkribus image upload requirements
(e.g. file size, image format) and not perform further preprocessing like binarisation:

```bash
python dd_preprocess.py path/to/source/directory path/to/destination/directory --basic_only
```

## Copyright

**dd_preprocess.py** was developed by Christina Sabbagh of SOAS University of London for the Divergent Discourses project. The project is a joint study involving SOAS University of London and Leipzig University, funded by the AHRC in the UK and the DFG in Germany.

Please acknowledge the project in any use of these materials. Copyright for the project resides with the two univerisities.
