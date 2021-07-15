# import the necessary packages
import sys 
sys.path.append("/home/deepernet/dev/Deblurring")
from preparing_results.aspectawarepreprocessor import AspectAwarePreprocessor
from preparing_results.metrics import PieAPP
import matplotlib.pyplot as plt
from sewar.full_ref import uqi
from sewar.full_ref import vifp
from sewar.full_ref import psnr
from sewar.full_ref import ssim
from imutils import paths
import numpy as np
import subprocess
import argparse
import csv
import cv2
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-gt", "--groundtruth", required=True, help="Path to the ground truth images")
# ap.add_argument("-res", "--results", required=True, help="Path to the deblurred images")
# ap.add_argument("-a", "--algorithm", required=True, help="name of the algorithm")
# args = ap.parse_args()

# DeblurGANv2
dataset_DeblurGANv2 = {
            "algorithm": "DeblurGANv2",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/DeblurGANv2/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGANv2/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGANv2/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGANv2/Lai_dataset/nonUniform"
                     ]
            }

# DeepBlur
dataset_DeepBlur = {
            "algorithm": "DeepBlur",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/DeepBlur/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeepBlur/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeepBlur/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeepBlur/Lai_dataset/nonUniform"
                     ]
            }

# SRN
dataset_SRN = {
            "algorithm": "SRN",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/SRN/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SRN/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SRN/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SRN/Lai_dataset/nonUniform"
                     ]
            }

# DeblurGAN
dataset_DeblurGAN = {
            "algorithm": "DeblurGAN",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/DeblurGAN/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGAN/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGAN/Lai/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DeblurGAN/Lai/nonUniform"
                     ]
            }

# SelfDeblur
dataset_SelfDeblur = {
             "algorithm": "SelfDeblur",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/SelfDeblur/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SelfDeblur/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SelfDeblur/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/SelfDeblur/Lai_dataset/nonUniform"
                     ]
            }


# DCP
dataset_DCP = {
            "algorithm": "DCP",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/DCP/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DCP/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DCP/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/DCP/Lai_dataset/nonUniform"
                     ]
            }

# ECP
dataset_ECP = {
            "algorithm": "ECP",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/ECP/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/ECP/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/ECP/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/ECP/Lai_dataset/nonUniform"
                     ]
            }

# Graph
dataset_Graph = {
            "algorithm": "Graph",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/Graph/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Graph/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Graph/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Graph/Lai_dataset/nonUniform"
                     ]
            }
# Two-phase
dataset_Two_phase = {
            "algorithm": "Two-phase",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "/home/deepernet/dev/Deblurring/Results/Two-phase/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Two-phase/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Two-phase/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/Two-phase/Lai_dataset/nonUniform"
                     ]
            }

# PMP
dataset_PMP = {
            "algorithm": "PMP",
            "GoPro": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/GoPro/groundtruths", 
                      "//home/deepernet/dev/Deblurring/Results/PMP/GoPro"
                     ], 
            "RealBlur": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/RealBlur/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/PMP/RealBlur"
                     ], 
            "Lai_uniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/PMP/Lai_dataset/uniform"
                     ], 
            "Lai_nonUniform": ["/home/deepernet/Data/Datasets/Deblurring/Evaluation_dataset/Lai/groundtruths", 
                         "/home/deepernet/dev/Deblurring/Results/PMP/Lai_dataset/nonUniform"
                     ]
            }

dataset_list = [dataset_DeblurGAN, dataset_DCP, dataset_ECP, dataset_DeblurGANv2, dataset_DeepBlur, 
                dataset_Graph, dataset_PMP, dataset_SelfDeblur, dataset_SRN, dataset_Two_phase]

def calc_metrics(imgA,imgB): 

    if imgA.shape != imgB.shape:
        asp = AspectAwarePreprocessor(300, 300)
        imgA = asp.preprocess(imgA)
        imgB = asp.preprocess(imgB)
    
    # Visual Information Fidelity
    vif_value = vifp(imgA, imgB)
    # Peak-to-noise-signal-ratio 
    psnr_value = psnr(imgA, imgB)
    # structural similaraty index 
    (ssim_value,cs_value) = ssim(imgA, imgB)


    return psnr_value, ssim_value, vif_value

def resize(imgA, imgB):
    r = 300
    asp = AspectAwarePreprocessor(r, r)
    imgA = asp.preprocess(imgA)
    imgB = asp.preprocess(imgB)

    # imgA = cv2.resize(imgA, (r, r), interpolation=cv2.INTER_AREA)
    # imgB = cv2.resize(imgB, (r, r), interpolation=cv2.INTER_AREA)
    return imgA, imgB

for dataset_dict in dataset_list: 
    
    for dataset_name in dataset_dict.keys(): 

        algorith = dataset_dict["algorithm"] 

        if dataset_name != "algorithm": 

            Heading = [f"{dataset_name} Dataset", "    ", "    ", "    ", "    "] 
            metrics = [" Image ", " PieAPP "," PSNR "," SSIM "," VIF "]

            # Grab the gt and results images 
            gt_imagepaths = list(paths.list_images(dataset_dict[dataset_name][0]))
            res_imagepaths = list(paths.list_images(dataset_dict[dataset_name][1]))

            with open(algorith + "_metrics.csv","a") as data_table: 
                writer = csv.writer(data_table) 
                writer.writerow(Heading)
                writer.writerow(metrics)

                for img_out in res_imagepaths: 
                    img_out_name = os.path.basename(img_out)
                    for img_truth in gt_imagepaths: 
                        img_truth_name = os.path.basename(img_truth)
                        if img_truth_name[:len(img_truth_name)-4] == img_out_name[:len(img_truth_name)-4]: 
                            imgA = cv2.imread(img_truth)
                            imgB = cv2.imread(img_out)

                            # Calculate PieAPP
                            cmd = ["python", f"/home/deepernet/dev/Deblurring/preparing_results/metrics/PieAPP/test_PieAPP_PT.py",
                                    "-ref", f"{img_truth}",
                                    "-out", f"{img_out}",
                                    "-gpu", "0"
                                    ]

                            completed_p2 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                            stdin=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                            encoding="utf-8")
                            PieAPP_metric, err = completed_p2.communicate()

                            # Calculate the metrics 
                            (PSNR, SSIM, VIF) = calc_metrics(imgA, imgB)

                            # store the values in the csv file 
                            metric_values = [img_truth_name[:len(img_truth_name)-4],PieAPP_metric, PSNR, SSIM, VIF]
                            writer.writerow(metric_values)
