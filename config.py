import os
import yaml

from seisnn.utils import make_dirs

SDS_ROOT = '/mnt/SDS_ROOT'
WORKSPACE = os.path.expanduser('~')


TFRECORD_ROOT = os.path.join(WORKSPACE, 'tfrecord')
DATASET_ROOT = os.path.join(TFRECORD_ROOT, 'dataset')
CATALOG_ROOT = os.path.join(TFRECORD_ROOT, 'catalog')
PICK_ROOT = os.path.join(TFRECORD_ROOT, 'picks')

MODELS_ROOT = os.path.join(WORKSPACE, 'models')
config = {'SDS_ROOT': SDS_ROOT,
          'WORKSPACE': WORKSPACE,
          'TFRECORD_ROOT': TFRECORD_ROOT,
          'DATASET_ROOT': DATASET_ROOT,
          'CATALOG_ROOT': CATALOG_ROOT,
          'PICK_ROOT': PICK_ROOT,
          'MODELS_ROOT': MODELS_ROOT,
          }

# mkdir for all folders and store into config.yaml
if __name__ == '__main__':
    for d in [TFRECORD_ROOT, DATASET_ROOT, PICK_ROOT, MODELS_ROOT]:
        make_dirs(d)

    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
