import sys
# add ASAP into python environment
sys.path.append('/opt/ASAP/bin')
from multiprocessing import Pool
from config import *
import multiresolutionimageinterface as mir




def generate_mask(tif,anno_path,mask_path,camelyon17_type_mask):
    reader = mir.MultiResolutionImageReader()
    basename = osp.basename(tif)
    output_path = osp.join(mask_path, basename)
    samplename=basename.rstrip('.tif')
    if osp.exists(output_path):
        return logging.info(f'{output_path} already exists, pass')
    mr_image = reader.open(tif)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(osp.join(anno_path, f'{samplename}.xml'))
    try:
        xml_repository.load()
    except:
        raise ValueError(f'check {samplename}.xml,it cannot be load!')
    annotation_mask = mir.AnnotationToMask()
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else ['_0', '_1', '_2']
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map,
                            conversion_order)
    logging.info(f'generate mask tif for {basename}')
    return None

if __name__=='__main__':
    pool = Pool(processes=10)
    for tif in sorted(train_tumor_tifs):
        # generate_mask(tif,train_tumor_anno,train_mask_path,camelyon17_type_mask)
        pool.apply_async(generate_mask,args=(tif,train_tumor_anno,train_mask_path,camelyon17_type_mask,))
    pool.close()
    pool.join()