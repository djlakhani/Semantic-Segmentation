## Semantic Segmentation

This project was completed for the course CSE 151B (Deep Learning) at UCSD.

The goal of semantic segmentation is to label each pixel in an image with the object it contains. We trained the models on the PASCAL VOC-2012 dataset which classifies image objects into one of 21 classes shown below:

```
class_mapping = [0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor]
```

The baseline model is a simple encoder, decoder FCN, with a weighted cross-entropy loss function and cosine annealing learning rate scheduler. In addition to the baseline, we experimented with a U-Net architecture and a ResNet-34 transfer learning model. The transfer learning model performed the best with pixel accuracy of 87% and the IoU of 0.46.

### Visualizing Results:

The transfer learning model produced the following results on the test dataset. The first image is the input image, the second is the true output mask, the third is the predicted output mask, and the last is the predicted output mask over the input image.

<img width="452" height="503" alt="image" src="https://github.com/user-attachments/assets/384abb3c-9ded-4be6-947c-114602c9db47" />

### Training and Validation Loss over Epochs for Transfer Learning Model:

### Files:

```best_transfer_model.pth```: Contains learned parameters for final transfer learning model.<br>
```train.py```: Training loop and related functions for baseline FCN training.<br>
```train_transfer.py```: Training loop and related functions for ResNet-34 Transfer model training.<br>
```basic_fcn.py```: Baseline model architecture.<br>
```transfer_fcn.py```: Transfer learning model architecture.<br>
```u-net.py```: U-Net architecture.<br>
```util.py```: Pixel accuracy and IoU function implementations.<br>
```voc.py```: Data processing and dataloader generator functions.<br>
```download.py```: Links to download dataset.<br>
