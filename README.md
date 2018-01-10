# meters

`Meters` the way to create machine learning models for recognition of digits on any meters.

Main features:
* Convert images to blosc format
* load images and labels (from csv, png and blosc)
* crop bbox, crop bbox to separate numbers

# About Meters

Meters has one module: [``batch``](https://github.com/analysiscenter/meters/tree/master/meters/batch).

``batch`` contains MeterBatch class witch include actions for preprocessing.

# Basic usage

Here is an example of pipeline that loads blosc images, make preprocessing and train a model over 50 epochs:
```python
ppl = (
	dataset.train
    .load(src=src, fmt='blosc', components='images')
    .load(src='./data/labels/meters.csv', \
    	  fmt='csv',\
          components='labels',\
          usecols=['file_name', 'counter_value'], crop_labels=True)
    .load(src='./data/labels/answers.csv', \
          fmt='csv', \
          components='coordinates',\
          usecols=['markup'])
	.normalize_images()
    .crop_to_bbox()
    .crop_to_numbers()
    .init_model('dynamic',
				MeterModel,
                'meter_model',
                config=model_config)
    .train_model('meter_model',
                 fetches='loss',
                 make_data=concatenate_water,
                 save_to=V('loss'),
                 mode='a')
    .run(batch_size=25, shuffle=True, drop_last=True, n_epochs=50)
)
```
# Installation

With [git clone](https://git-scm.com/docs/git-clone):

	git clone https://github.com/analysiscenter/meters.git