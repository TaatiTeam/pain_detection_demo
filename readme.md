A minimal repository to demo the pain detection model proposed in the paper titled _Unobtrusive Pain Monitoring in Older Adults with Dementia
using Pairwise and Contrastive Training_. [Link will be added once published]

<p align="center"><img src="docs/images/saliency-S64-wint-contrastive.png" /></p>

The code is tested on Linux with Python 3.6+, PyTorch 1.6, and Cuda 10.2

After installing the requirements, you should be able to run `test.py`.
It should print out 5.49 pain score (PSPI) for the frames in the `example_frames` folder.

`test.py` will also do a “frame rate test” and print out how many frames per second your system is capable of processing.
We achieved ~9FPS on a PC with an NVIDIA RTX-2080 Ti GPU and Intel i9-9900K CPU @ 3.60GHz.
Currently [Face Alignment Network (and S3FD)](https://github.com/1adrianb/face-alignment) are use to detect and align faces.
These could be swapped for faster non-deep learning methods to improve performance.

