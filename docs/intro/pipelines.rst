Pipelines
==========

This module contains a class that allows to create a pipeline which you can use to train models and make predictions. You can configurate pipelines the way you want and use them on your data or adjust pipelines in order to get better perfomance.

How to use
-----------
Workflow consists of 4 steps. First, factory class is imported::

	from meters.pipelines import PipelineFactory

Second, an instance of ```PipelineFacory``` class is created::

	ppl_class = PipelineFactory()

Third, pipeline is designed. This step includes specifications for:

1. path to the data (``src``)

2. the shape (``shape``) of the input images (original images will be resized if their shape is different)

3. model and its training procedure (ppl_config)::
	
	src = 'path/to/data'
	train_ppl = ppl_class.simple_train(src, shape, ppl_config={'model': VGG19, 'model_name': 'vgg', n_epochs: 100, batch_size: 25})

Fourth, dataset is fed to the pipeline and calculations are performed::

	(dset.train >> train_ppl).run()

Result is a trained model or values stored in pipeline variables (e.g. loss).

Available functions
--------------------
consists of following pipelines:

* load_all
* make_digits
* simple_train
* simple_predict

API
----
See :doc:`Pipeline class API <../api/meters.pipelines>`
