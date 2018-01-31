Batch
======

This module stores batch class for preprocessing meters images.

MeterBatch
-----------

`MeterBatch` is the main class that defines how to store images and contains all actions for preprocessing.

Attributes of MeterBatch:

* ``images``, input images
* ``labels``, targets for images - array of strings with numbers
* ``coordinates``, array with four numbers - coordinates of the top-left corner, height and width of the bounding box
* ``display``, array with cropped bounding boxes' areas from images
* ``digits``, array with ``num_split`` numbers from the meter.

Actions of MeterBatch allows to:

* load images from blosc formats and labels from csv format
* crop bounding box's area from an image
* split meter's value to separate digits

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
        .load(src='path/to/data.csv', fmt='csv', components=['coordinates', 'labels'], index_col='file_name')
        .crop_from_bbox()
        .split_labels()
        .split_to_digits()
    )

    ppl = (template_ppl << dset)
    batch = ppl.next_batch(25)

API
---
See :doc:`Batch API <../api/meters.batch>`
