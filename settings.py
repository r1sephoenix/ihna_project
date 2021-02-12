import os
import re
from dataclasses import field, dataclass, InitVar
from typing import List, Dict


@dataclass
class EEGSettings:
    """
    Settings for EEG postprocessing

    Constructor arguments:
        channels_to_drop: names of channels to drop
        montage_name: name of EEG equipment montage
        events: names of events
        fr_bands: frequency bands for obtaining features
        dict_cls: indexes for pairwise classification
        mat_indexes: indexes of mat group subjects
        not_mat_indexes: indexes of not mat group subjects
    """
    channels_to_drop: List[str] = field(default_factory=lambda: ['E8', 'E14', 'E21', 'E25', 'E43', 'E48', 'E49', 'E56',
                                                                 'E57', 'E63', 'E64', 'E65', 'E68', 'E69', 'E73', 'E74',
                                                                 'E81', 'E82', 'E88', 'E89', 'E90', 'E94', 'E95', 'E99',
                                                                 'E100', 'E107', 'E113', 'E119', 'E120', 'E125', 'E126',
                                                                 'E127', 'E128', 'Status'])
    montage_name: str = 'GSN-HydroCel-128'
    events: List[int] = field(default_factory=lambda: [241, 242, 244])
    fr_bands: Dict[str, List[int]] = field(default_factory=lambda: {'theta1': [4, 6],
                                                                    'theta2': [6, 8],
                                                                    'alpha1': [8, 10],
                                                                    'alpha2': [10, 12],
                                                                    'beta1': [12, 16],
                                                                    'beta2': [16, 20],
                                                                    'beta3': [20, 24]})
    dict_cls: Dict[str, List[int]] = field(default_factory=lambda: {'241/244': [0, 2],
                                                                    '242/244': [1, 2],
                                                                    '241/242': [0, 1]})
    subj_indexes: List[str] = field(default_factory=list)
    mat_indexes: List[str] = field(default_factory=lambda: ['311', '312', '314', '315', '316', '317', '326', '327',
                                                            '328', '330', '334', '335'])
    not_mat_indexes: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    path: InitVar[str] = ''

    def __post_init__(self, path):
        if path == '':
            raise ValueError('Please type path to EEG files folder')
        files = [f for f in sorted(os.listdir(path))]
        _ = files.pop(0)
        self.files = files
        indexes = [re.search('(.+?)_', file_name).group(1) for file_name in files]
        self.indexes = indexes
        self.not_mat_indexes = [x for x in indexes if x not in self.mat_indexes]
