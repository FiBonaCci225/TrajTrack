import yaml
import os
import os.path as osp
import glob
import numpy as np
from easydict import EasyDict
from .utils import recreate_dirs


class Config:

    def __init__(self, cfg_id, tmp=False, create_dirs=False):
        self.id = cfg_id
        cfg_path = 'configs/%s.yaml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        self.yml_dict = EasyDict(yaml.safe_load(open(files[0], 'r')))


    def __getattribute__(self, name):
        yml_dict = super().__getattribute__('yml_dict')
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__('yml_dict')
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
            