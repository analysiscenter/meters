# meters

`Meters` the way to create machine learning models for recognition of digits on any meters.

Main features:
* Convert images to blosc format
* load images and labels (from csv, png and blosc)
* crop bbox, crop bbox to separate numbers

# About Meters

Meters has one module: (``batch``)[https://github.com/analysiscenter/meters/tree/master/meters/batch].

``batch`` contains MeterBatch class witch include actions for preprocessing.

# Basic usage

Here is an example of pipeline that loads blosc images and make preprocessing:
```python
ppl = Pipeline()
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
```
# Installation
