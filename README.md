# Comprehensive Evaluation of Blind Single Image Deblurring Algorithms
## **Introduction** 
This project covers the work completed in a comparative research study which evaluates blind single image deblurring algorithms on real world blurs 

## **Datasets**
1.  Lai et al. [Real and synthetic dataset](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/)
2.  GoPro [dataset](https://seungjunnah.github.io/Datasets/gopro.html)
3. RealBlur [dataset](http://cg.postech.ac.kr/research/realblur/) 

Sampled dataset: Evaluation [dataset](https://drive.google.com/drive/folders/1_WvvK3WSSgWzjbOKIV1k3xg2JDSC2qub?usp=sharing)

## **Metrics**
1. PieAPP metric: https://github.com/prashnani/PerceptualImageError
2. Liu metric: https://gfx.cs.princeton.edu/pubs/Liu_2013_ANM/index.php
3. PSNR
4. SSIM 
5. VIF  
## **Evaluation scripts**
### compute_metrics.py 
* Computes the evaluation metrics and save them in csv files

### aspectawarepreprocessor.py
* Resizes groundtruth and blurred image pairs to the same size. 

### draw_boxplots.py 
* Visualizes the boxplots of the algorithms
