Pipelines
==========

This module contains a class that allows to create a pipeline which we can use to train models and make predictions. You can configurate pipelines the way you want and use them on your data or adjust pipelines in order to get better perfomance.

How to use
-----------
Working with pipelines consists of 4 steps. First, we import class with all pipelines::

	from meters.pipelines import PipelineFactory

Second, we create an instance of the ```PipelineFacory``` class::

	ppl_class = PipelineFactory()

Third, we design a pipeline(e.g. simple_train), add the data path to src, if you want, resizing the images by adding ``shape`` parameter and specify its parameters(e.g. path to data, model, model name, n_epochs and batch_size)::
	
	src = 'path/to/data'
	train_ppl = ppl_class.simple_train(src, shape, ppl_config={'model': VGG19, 'model_name': 'vgg', n_epochs: 100, batch_size: 25})

Fourth, we pass dataset to the pipeline and run caclulation::

	(dset.train >> train_ppl).run()

Result is typically a trained model or some values stored in pipeline variables (e.g. loss).

Available functions
--------------------
At this moment the class contains following pipelines:

* load_all
* make_digits
* simple_train
* simple_predict

API
----
See :doc:`Pipeline class API <../api/meters.pipelines>`
