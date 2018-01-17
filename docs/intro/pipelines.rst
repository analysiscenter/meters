Pipelines
==========

This module contains a calss which allows to create a pipeline that we used to train models and make predictions. You can configurate pipelines your way and use it on own data or or adjust pipelines in order to get better perfomance.

How to use
-----------
Working with pipelines consists of 4 steps. First, we import class with all pipelines::

	from meters.pipelines import PipelineFactory

Second, we create the onstance of the ```PipelineFacory``` class::

	ppl_class = PipelineFactory()

Third, we desired pipeline(e.g. simple_train) and specify its parameters(e.g. path to data, model, model name, n_epochs and batch_size)::

	train_ppl = ppl_class..simple_train(src, ppl_config={'model': VGG19, 'model_name': 'vgg', n_epochs: 100, batch_size: 25})

Fourth, we pass dataset to the pipeline and run caclulation::

	(dset.train >> train_ppl).run()

Result is typically a trained model or some values stored in pipeline variables (e.g. loss).

Available functions
--------------------
At this moment the class contains following pipelines:

* load_all
* crop_digits
* simple_train
* simple_predict

API
----
See :doc:`Pipeline class API <../api/meters.pipelines>`
