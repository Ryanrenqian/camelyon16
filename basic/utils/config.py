import json
from basic.data import *
class Config:
    def __init__(self,configfile):
        f =open(configfile,'r')
        try:
            config = json.load(f)
        except:
            raise ValueError('please check your config file')
        self.config=config
        self._get_data()
        self._update_config()

    def _get_data(self):
        '''
        如果存在已有的数据集,则直接调入,不存在则重新生成新的数据集
        :return:
        '''
        try:
            dataset = self.config["dataset"]["train"]["save_path"]
            if self.config['predata']['method']=="Mask":
                type="Mask"
            elif self.config['predata']['method']=="Label":
                type = "Label"
        except:
            dataset = os.path.join(self.config['workspace'], 'patch_list')
            self.config["dataset"]["train"]["save_path"] = dataset
            self.generate = GenerateData(**self.config)
            os.system(f'mkdir -p {dataset}')
            if self.config['predata']['method']=="Mask":
                type="Mask"
                num,maskfile=self.generate.preMask()
                self.config['dataset']["train"]['samplelist'] = maskfile
                logging.info(f'generate {num} mask and patches in {self.config["dataset"]["train"]["samplelist"]}')
            elif self.config['predata']['method']=="Label":
                type = "Label"
                tumor_list, normal_list = self.generate.preLimited(datasize=400000)
                self.config["dataset"]["train"]['tumor_list'] = tumor_list
                self.config["dataset"]["train"]['normal_list'] = normal_list
            self.config["dataset"]['type'] = type

    def _update_config(self):
        configpath=os.path.join(self.config['workspace'],'config.json')
        with open(configpath,'w')as f:
            json.dump(self.config,f,indent='\t')