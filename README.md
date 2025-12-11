# STAT605 PROJECT

Clone and install dependencies:

```bash
pip install opencv-python numpy torch torchvision segmentation-models-pytorch tqdm pandas
```

Or use requirements file:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python inference_test.py
```

Check the `samples` folder to see the results.

Run on custom image:

```bash
python inference_test.py path/to/image.png
```

Output files are saved in the same directory as the input image.
