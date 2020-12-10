"""SDSS dataset."""

import tensorflow_datasets as tfds
from astropy.io import fits
import json
import tensorflow as tf
import os

# TODO(SDSS): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
selected features from SDSS dataset in 

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
            'flux': tfds.features.Tensor(shape=(None,1),dtype=tf.float32),
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
            "local_dir": '/global/cscratch1/sd/vboehm/Datasets/SDSS_BOSS_data/',
            "data_dir": '/global/project/projectdirs/cosmo/data/sdss/dr16/sdss/spectro/redux/v5_13_0/'
            },
        ),
    ]

  def _generate_examples(self, local_dir, data_dir):
    """Yields examples."""
    with open(os.path.join(local_dir,'datafiles.txt'), 'r') as infile:
        for line in infile:
            filenames = json.loads(line)
    infile.close()

    for jj, data_file in enumerate(filenames[0:10]):
        print(data_file)
        with tf.io.gfile.GFile(os.path.join(data_dir ,data_file), mode='rb') as f:
            #flux = 1*fits.getdata(f, 0).astype('float32')
            hdulist = fits.open(f)
            flux    = hdulist[0].data.astype('float32')
            for ii in range(1000):
                spec = tf.expand_dims(flux[ii],-1)
                yield '%d'%int(jj*1000+ii), {'flux': spec.numpy()}#, "label":label}
            f.close()
