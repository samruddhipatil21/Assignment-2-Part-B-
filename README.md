# Assignment-2-Part-B-
## Part B
PyTorch, PyTorch Lightning framework to train a fine-tuned **ResNet18 model** 
### Instructions to train and evaluate the model:
1. Install the required libraries:
```python
!pip install pytorch_lightning
!curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > Asg2_Dataset.zip
!unzip Asg2_Dataset.zip
!pip install wandb
```
2. Give path for the dataset.
3. To fine-tune pre-trained model, use the following command
```python
net = FineTuneTask(10)
trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=1)
trainer.fit(model=net,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
```
4. To evaluate the model, use the following command:
```python
trainer.test(net,test_dataloader)
```
### Dataset and Data Loaders
The iNaturalist 12K dataset is loaded from the \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/train  and \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/val \

The dataset is split into training and validation sets using a ratio of 80:20. 
    
### Data Transformations
Two sets of data transformations are specified: 'transform' and 'transform_augmented'.
Both resize the images to 256x256 pixels and convert them into tensors. However, 'transform_augmented' incorporates additional data augmentation methods, such as random cropping, flipping, and rotating.
After these operations, both transformations normalize the images using the mean and standard deviation values from the ImageNet dataset.

### Methods
The __init__ function initializes a ResNet18 model with a modified output layer containing num_classes neurons. In the forward function, the model defines how input tensors are processed.

The training_step function manages a single training step, where it computes the loss and accuracy of the model on a batch of training data and records these values.

Similarly, the validation_step function computes the loss and accuracy during validation and logs the results. Lastly, the test_step function calculates the loss and accuracy of the model during testing.


