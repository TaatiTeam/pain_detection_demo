A minimal repository to demo the pain detection model proposed in the paper titled [Unobtrusive Pain Monitoring in Older Adults with Dementia
using Pairwise and Contrastive Training](https://ieeexplore.ieee.org/document/9298886). It is available under the "Early Access" area on IEEE Xplore, and it can be cited with its DOI.

<p align="center"><img src="docs/images/saliency-S64-wint-contrastive.png" /></p>

The code is tested on Linux with Python 3.6+, PyTorch 1.6, and Cuda 10.2

After installing the requirements, you should be able to run `test.py`, and it should print out the pain score (PSPI) for the frames in the `example_frames` folder.

Two pretrained models are included. One was trained on the UNBC-McMaster _Shoulder Pain Expression Archive_ dataset **and** the University of Regina's _Pain in Severe Dementia_ dataset.
And another checkpoint that was trained on the UNBC-McMaster dataset **only**. In both cases, UNBC subjects 66, 80, 97, 108, and 121 were excluded from training.


`test.py` can also do a “frame rate test” and print out how many frames per second your system is capable of processing.
We achieved ~9FPS on a PC with an NVIDIA RTX-2080 Ti GPU and Intel i9-9900K CPU @ 3.60GHz.
Currently [Face Alignment Network (and S3FD)](https://github.com/1adrianb/face-alignment) are used to detect and align faces.
These could be swapped for faster non-deep learning methods to improve performance.




