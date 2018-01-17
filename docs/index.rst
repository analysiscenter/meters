==================================
Welcome to Meter's documentation!
==================================
`Meters` is a framework for building end-to-end machine learning models for recognition of digits on any meters.

Main features:

* convert images to blosc format
* load images(from blosc) and labels (from csv)
* crop bbox, split bbox and labels to separate numbers
* arrange new custom actions into pipeline
* easily configure pipeline
* build, train and test custom models for research.

.. note:: Meters is based on `Dataset <https://github.com/analysiscenter/dataset>`_. You may benefit from reading `its documentation <https://analysiscenter.github.io/dataset>`_. However, it is not required, especially in the beginning.

Meters has two module: :doc:`batch <./api/meters.batch>` and :doc:`pipelines <./api/meters.pipelines>`

``batch`` module contains MeterBatch class witch includes actions for preprocessing.

``pipelines`` module contains PipelineFactory class witch builded pipelines for:

* tarin simple model to classify digits on meters
* predict on pretrained model

Contents
=========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   api/meters

Basic usage
============
Here is an example of pipelines that loads blosc images, makes preprocessing and trains a model for 50 epochs::

  ppl = (
      dataset.train
      .load(src=src, fmt='blosc', components='images')
      .load(src='path/to/labels', fmt='csv', components='labels', index_col='file_name')
      .load(src='path/to/coordinates', fmt='csv', components='coordinates', index_col='file_name')
      .normalize_images()
      .crop_to_bbox()
      .crop_to_digits()
      .crop_labels()
      .init_model('dynamic', MeterModel, 'meter_model', config=model_config)
      .train_model('meter_model', fetches='loss', make_data=concatenate_water, save_to=V('loss'), mode='a')
      .run(batch_size=25, shuffle=True, drop_last=True, n_epochs=50)
  )

Installation
=============
With `git clone <https://git-scm.com/docs/git-clone/>`_::

    git clone https://github.com/analysiscenter/meters.git

If your python file is located in another directory, you might need to add a path to `meters` submodule location::

    import sys
    sys.path.insert(0, '/path/to/meters')
    import meters

Citing Meters
==============
Please cite Meters in your publications if it helps your research.:

``Khudorozhkov R., Broilovskiy A., Mylzenova D., Ivanov G. Meters library for data science research of meters. 2018.``
