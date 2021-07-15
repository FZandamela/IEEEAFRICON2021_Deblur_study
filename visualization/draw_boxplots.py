import os
import pandas as pd
import matplotlib.pyplot as plt



# data_dir = r"E:\trats\CloudStorarge\OneDrive - csir.co.za\Image Deblurring\Part 1 - Evaluation methods and metrics\Evaluation\Visualization"
data_dir = "/home/deepernet/dev/Deblurring/Results/visualization/Visualization"
fname_dict = {
    "GoPro": "Evaluation_merics_22_04_2021_GoPro_Dataset.xlsx"
    ,"Lai Real": "Evaluation_metrics_Liu.xlsx"
    ,"Lai synthetic nonuniform": "Evaluation_merics_22_04_2021_Lai_synthetic_nonuniform_Dataset.xlsx"
    ,"Lai synthetic uniform": "Evaluation_merics_22_04_2021_Lai_synthetic_uniform_Dataset.xlsx"
    ,"RealBlur": "Evaluation_merics_22_04_2021_RealBlur_Dataset.xlsx"
    }


""" Compare algorithms on different datasets. A graph should have one metrics, different algorithms and one dataset.
"""
for dataset_name, fname in fname_dict.items():
    print(f"Processing: {dataset_name}")
    
    fname_path = os.path.join(data_dir, fname)

    # read all the sheets.
    df = pd.read_excel(fname_path, sheet_name=None)



    # PSNR metric
    psnr = []
    psnr_colname = []

    # SSIM
    ssim = []
    ssim_colname = []


    # VIF
    vif = []
    vif_colname = []

    # PieApp
    pieapp = []
    pieapp_colname = []

    # Real: Lai metric
    real_lai = []
    real_lai_colname = []




    use_PSNR, use_SSIM, use_VIF, use_PieAPP, use_liu = False, False, False, False, False

    for alg_name, df_data in df.items():
        if len(df_data) == 0:
            print(f'No data for for algorithm: {alg_name} in file {fname}')
            continue
        
        # clean the column names
        df_data.columns = list(map(str.strip, list(df_data.columns)))

        #PSNR
        if 'PSNR' in df_data.columns:
            psnr.append(df_data['PSNR'])
            psnr_colname.append(alg_name)
            use_PSNR = True

        # SSIM
        if 'SSIM' in df_data.columns:
            ssim.append(df_data['SSIM'])
            ssim_colname.append(alg_name)
            use_SSIM = True

        # VIF
        if 'VIF' in df_data.columns:
            vif.append(df_data['VIF'])
            vif_colname.append(alg_name)
            use_VIF = True

        # PieApp
        if 'PieAPP' in df_data.columns:
            pieapp.append(df_data['PieAPP'])
            pieapp_colname.append(alg_name)
            use_PieAPP = True
        
        if 'Liu Metric' in df_data.columns:
            real_lai.append(df_data['Liu Metric'])
            real_lai_colname.append(alg_name)
            use_liu = True

    
    # create a  dataframe
    # PSNR




    # ----
    show_grid = False
    figsize = (12,6)

    # PSNR
    if use_PSNR:
        df_psnr = pd.concat(psnr, ignore_index=True, axis=1)
        df_psnr.columns = psnr_colname
        
        plt.figure(1, figsize=figsize)
        plt.clf()
        bp_psnr = df_psnr.boxplot(grid=show_grid)
        bp_psnr.set_title( f"{dataset_name}: PSNR", fontweight="bold")
        # bp_psnr.axes.set_ylim(0, 25)
        bp_psnr.figure.savefig(f"code/results/{dataset_name}_psnr.png")

    # SSIM
    if use_SSIM:
        df_ssim = pd.concat(ssim, ignore_index=True, axis=1)
        df_ssim.columns = ssim_colname

        plt.figure(2, figsize=figsize)
        plt.clf()
        bp_ssim = df_ssim.boxplot(grid=show_grid)
        bp_ssim.set_title(f"{dataset_name}: SSIM", fontweight="bold")
        # bp_ssim.axes.set_ylim(0, 1)
        bp_ssim.figure.savefig(f"code/results/{dataset_name}_ssim.png")


    # # VIF
    if use_VIF:
        df_vif = pd.concat(vif, ignore_index=True, axis=1)
        df_vif.columns = vif_colname

        plt.figure(3, figsize=figsize)
        plt.clf()
        bp_vif = df_vif.boxplot(grid=show_grid)
        bp_vif.set_title(f"{dataset_name}: VIF", fontweight="bold")
        # bp_vif.axes.set_ylim(0, 0.25)
        bp_vif.figure.savefig(f"code/results/{dataset_name}_vif.png")

    
    # PieAPP
    if use_PieAPP:
        df_pieapp = pd.concat(pieapp, ignore_index=True, axis=1)
        df_pieapp.columns = pieapp_colname
    
        plt.figure(4, figsize=figsize)
        plt.clf()
        bp_pieapp = df_pieapp.boxplot(grid=show_grid)
        bp_pieapp.set_title(f"{dataset_name}: PieAPP", fontweight="bold")
        # bp_vif.axes.set_ylim(0, 0.25)
        bp_pieapp.figure.savefig(f"code/results/{dataset_name}_pieapp.png")
    
    if use_liu:
        df_real = pd.concat(real_lai, ignore_index=True, axis=1)
        df_real.columns = real_lai_colname

        plt.figure(5, figsize=figsize)
        bp_real = df_real.boxplot(grid=show_grid)
        bp_real.set_title(f"{dataset_name}: Liu Metric", fontweight="bold")      #changed Lia to Liu
        bp_real.figure.savefig(f"code/results/{dataset_name}_liu_metric.png")

    # plt.show()