# Edge-Based-Image-Restoration
This is the code of Edge-Based Image Restoration, implemented by lhaof.

If you use this code, you need to cite the following paper
```
@article{rares2005edge-based,
  author = {Rares, Andrei and Reinders, Marcel and Biemond, Jan},
  year = {2005},
  month = {01},
  pages = {1454-1468},
  title = {Edge-Based Image Restoration.},
  volume = {14},
  journal = {IEEE Transactions on Image Processing}
}
```
I implemented this code when I was submitting my paper 'CASI: Context-Aware Semantic Inpainting'. Because I was required to compare my work with this algorithm 'Edge-Based Image Restoration'. Anyway, you may kindly cite my work below
```
@article{li2018context-aware,
  title={Context-Aware Semantic Inpainting},
  author={Li, Haofeng and Li, Guanbin and Lin, Liang and Yu, Hongchuan and Yu, Yizhou},
  journal={IEEE Transactions on Systems, Man, and Cybernetics},
  pages={1--14},
  year={2018}
}
```
As I remember, 'imrestore.py' works for gray-scale images while 'rgbrestore.py' deals with RGB images.
You need to install
```
Python
Matlab
matlab.engine for python (then you can call matlab function within a python script.)
```
