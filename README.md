# Logistic regression model

This is a made from scratch logistic regression model based on the course Neural Networks and Deep learning by Andrew Ng.

The dataset used to train and test the model was gathered from Kaggle ([Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog))

## Technical details

The current version of this repo's algorithm is its first iteration. Therefore, a lot of improvements can and will be made (ToDo...).

If you want to fork it and run your own train/test sets, be aware of the params in load_imgs and train functions
* load_imgs
  * load_train_set -> set it to true (default) if you want to load the train image set, otherwise will load the test_set
  * image_limit -> this limits the amount of images to load per category. if it is set to 200, in this case, it will load 200 cat images and 200 dog images
 
* train
  * learning_rate -> (default = 0.0005) adjust it for more accurate changes in the training process (smaller values, slow improvements; higher values, more risk of dancing cost values)
  * num_iterations -> (default = 10000) number of iterations you want the train process to handle.

## Running steps

```
python3 model.py
```



