# camelyon16 
### train model with config file
```bash
python bin/main.py -c bin/config
```
This script includes *Three* steps:
1. extract samples based on OTSU, ground truth mask, original image files.
in `bin/config.json` `dataset/train`, parameters for generation of TrainSet is given.
2. train your model, you can appoint model you used in `bin/config` file(only include `Scannet` now)

#### detailed about config file
`"workspace":` :where you save your data 
`"model":` :which model you used. make sure your models available in `models\_init_.py`. now only scannet was included.

`"dataset":{}`: parameters for dataset, you can change value of `threshold` in `dataset` to extract different numbers of samples.
`threshold` is based on OTSU dense in patches. because in the posttreatment such an threshold was used for fast filter non-information area.
`gt_mask_folder`:gt_mask_folder generated from `wsi_prepare/wsi_mask.py`.`otsu_folder`:otsu_folder generated from `wsi_prepare/wsi_otsu.py`
**warning**: `resize` is the downsamle value you used in `wsi_prepare/wsi_mask.py` and `wsi_prepare/wsi_otsu.py`. and their values should be same.
    
`"train":{}` :parameters for train stage.
`"valid":{}`:parameters for train valid.
`"hard":{}`:parameters for hard-mining stage.

### Segmentaiton
After training  model, you can use scripts in `posttreatment` for segmentation

### Analysis 

 