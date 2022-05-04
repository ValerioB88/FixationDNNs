import numpy as np
import torch
import neptune.new as neptune
from sty import ef, rs, fg, bg

class Config:
    def __init__(self, **kwargs):
        self.use_cuda = torch.cuda.is_available()
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]
        self.finalize_init(**kwargs)

    def __setattr__(self, *args, **kwargs):
        if hasattr(self, 'weblogger'):
            if isinstance(self.weblogger, neptune.Run):
                self.weblogger[f"parameters/{args[0]}"] = str(args[1])
        super().__setattr__(*args, **kwargs)

    def finalize_init(self, list_tags=None, **PARAMS):
        if list_tags is None:
            list_tags = []
        print(fg.magenta)
        print('**LIST_TAGS**:')
        print(list_tags)
        if self.verbose:
            print('***PARAMS***')
            if not self.use_cuda:
                list_tags.append('LOCALTEST')

            for i in sorted(PARAMS.keys()):
                print(f'\t{i} : ' + ef.inverse + f'{PARAMS[i]}' + rs.inverse)

        if self.weblogger == 2:
            neptune_run = neptune.init(f'valeriobiscione/{self.project_name}',
                                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4N2Y2NzVkYS04NDYzLTQ2MjQtYTBmYS1hZGI1NzFmZjcwNzIifQ==')
            neptune_run["sys/tags"].add(list_tags)
            neptune_run["parameters"] = PARAMS
            self.weblogger = neptune_run
        print(rs.fg)
