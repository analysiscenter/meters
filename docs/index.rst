==================================
Welcome to Meter's documentation!
==================================
`Meters` is a framework for building end-to-end machine learning models for recognition of digits on any meters.

Main features:

* convert images to blosc format
* load images(from blosc) and labels (from csv)
* separate numbers (and labels respectively) from cropped area with meter's value
* arrange new custom actions into pipeline
* easily configure pipeline
* build, train and test custom models for research.

.. note:: Meters is based on `Dataset <https://github.com/analysiscenter/dataset>`_. You may benefit from reading `its documentation <https://analysiscenter.github.io/dataset>`_. However, it is not required, especially in the beginning.

Meters has two module: :doc:`batch <./api/meters.batch>` and :doc:`pipelines <./api/meters.pipelines>`

``batch`` module contains MeterBatch class which includes actions for preprocessing.

``pipelines`` module contains PipelineFactory class that builds pipelines for:

  * training simple models to classify digits on meters
  * making predictions by trained model.

Contents
=========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   api/meters

Basic usage
============
Here is an example of a pipeline that loads blosc images, makes preprocessing and trains a model for 50 epochs::

  ppl = (
      dataset.train
      .load(src=src, fmt='blosc', components='images')
      .load(src='path/to/data.csv', fmt='csv', components=['coordinates', 'labels'], index_col='file_name')
      .crop_from_bbox()
      .split_labels()
      .split_to_digits()
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
Please cite Meters in your publications if it helps your research:

``Khudorozhkov R., Broilovskiy A., Mylzenova D., Ivanov G. Meters library for data science research of meters. 2018.``
