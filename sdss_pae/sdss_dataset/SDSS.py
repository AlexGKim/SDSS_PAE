"""SDSS dataset."""

import tensorflow_datasets as tfds
from astropy.io import fits
import tensorflow as tf
import os
import numpy as np
            #f.close()
            #f.close()

# TODO(SDSS): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(SDSS): BibTeX citation
_CITATION = """
"""


class Sdss(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SDSS dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(SDSS): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Tensor(shape=(4000,1),dtype=tf.float32),
        #    'label': tfds.features.ClassLabel(names=['STAR', 'QSR', 'GALAXY']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,#('flux'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
            "data_dir": '/global/cscratch1/sd/vboehm/Datasets/SDSS_BOSS_data/',
            },
        ),
    ]

  def _generate_examples(self, data_dir):
    """Yields examples."""
    print('begin')
    # Read the maps from the directory
    for ii, image_file in enumerate(tf.io.gfile.listdir(data_dir)):
        print(os.path.join(data_dir ,image_file))
        with tf.io.gfile.GFile(os.path.join(data_dir ,image_file), mode='rb') as f:
            data = 1*fits.getdata(f, 0).astype('float32')
            #for ii in range(1000):
            im = tf.expand_dims(data[0][0:4000],-1)
            print(im)
            f.close()
            yield '%d'%int(ii), {'image': im.numpy()}#, "label":label}
