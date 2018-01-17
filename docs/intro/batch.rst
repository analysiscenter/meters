Batch
======

This module stores batch class for preprocessing meters images.

MeterBatch
-----------

`MeterBatch` is the main class that define how to store images and contains all action to preprocessing.

Attributes of MeterBatch:

* ``images``, input images
* ``labels``, answer for images - string with numbers
* ``coordinates``, array with four numbers - coordinates of one of the corners of the bbox, height and width
* ``cropped``, array with cropped image by ``coordinates``
* ``sepcrop``, array with ``num_split`` numbers from the meter.

Actions of MeterBatch allows e.g.:

* load images from blosc formats and labels from csv format
* crop images to bbox
* split bbox on separate numbers
* normalize images

Usage
-----

If you want to work with MeterBatch you need to create a pipeline object::

    from meters import MeterBatch
    import dataset, FilesIndex, Pipeline

    ix = FilesIndex(path='path/to/images', no_ext=True)
    dset = Dataset(fileindex, batch_class=MeterBatch)

    template_ppl = (
        Pipeline()
        .load(src=src, fmt='blosc', components='images')
        .load(src='path/to/labels', fmt='csv', components='labels', index_col='file_name'))
        .load(src='path/to/coordinates', fmt='csv', components='coordinates', index_col='file_name')
        .normalize_images()
        .crop_to_bbox()
        .crop_to_digits()
        .crop_labels()
    )

    ppl = (template_ppl << dset)
    batch = ppl.next_batch(25)

API
---
See :doc:`Batch API <../api/meters.batch>`