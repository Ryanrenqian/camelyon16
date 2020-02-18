import os.path as osp
import os
import logging
import glob
logging.basicConfig(level=logging.INFO)
# only one thing u need to do is setting variable between two comments
camelyon16_data_path="/root/workspace/dataset/CAMELYON16/"
camelyon17_type_mask = False


train_tumor_tif = osp.join(camelyon16_data_path,'training','tumor')
train_tumor_anno = osp.join(camelyon16_data_path,'training','lesion_annotations')
logging.info(f'train_tumor_anno in{train_tumor_anno}')
logging.info(f'train_tumor_tif in {train_tumor_tif}')
logging.info('coverting tumor annotation into Mask image as tif file')
train_mask_path = osp.join(camelyon16_data_path,'training','mask')
if not osp.exists(train_mask_path):
    os.mkdir(train_mask_path)
    logging.debug(f'{train_mask_path} doesn\'t exists and is created automatic')
train_tumor_tifs = glob.glob(f'{train_tumor_tif}/*.tif')


