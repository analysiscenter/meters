==================================
Welcome to Meter's documentation!
==================================
`Meters` is a framework for building end-to-end machine learning models for recognition of digits on any meters.

Main features:

* convert images to blosc format
* load images and labels (from csv, jpg and blosc
* crop bbox, split bbox and labels to separate numbers
* arrange new custom actions into pipeline
* easily configurate pipeline
* build, train and test custom models for deep research.

.. note:: Meters is based on `Dataset <https://github.com/analysiscenter/dataset>`_. You might benefit from reading `its documentation <https://analysiscenter.github.io/dataset>`_. However, it is not required, especially at the beginning.

Meters has one module: :doc:`batch <./api/meters.batch>`

``batch`` contains MeterBatch class witch include actions for preprocessing.

Contents
=========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   api/meters

Basic usage
============
Here is an example of pipeline that loads blosc images, make preprocessing and train a model over 50 epochs::

  ppl = (
      dataset.train
      .load(src=src, fmt='blosc', components='images')
      .load(src='data/labels/meters.csv', fmt='csv', components='labels',
            usecols=['file_name', 'counter_value'], crop_labels=True)
      .load(src='data/labels/answers.csv', fmt='csv', components='coordinates',
            usecols=['markup'])
      .normalize_images()
      .crop_to_bbox()
      .crop_to_numbers()
      .init_model('dynamic', MeterModel, 'meter_model', config=model_config)
      .train_model('meter_model', fetches='loss', make_data=concatenate_water,
                   save_to=V('loss'), mode='a')
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
Please cite Meters in your publications if it helps your research.::

    Khudorozhkov R., Broilovskiy A., Mylzenova D., Ivanov G. Meters library for data science research of heart signals. 2018.
