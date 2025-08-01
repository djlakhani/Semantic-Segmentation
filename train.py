from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing

num_workers = multiprocessing.cpu_count()

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():
    # TODO for Q4.c || Caculate the weights for the classes
    raise NotImplementedError

# normalize using imagenet averages
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
val_dataset_new_len = int(0.8 * len(val_dataset))
test_dataset_len = len(val_dataset) - val_dataset_new_len
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_dataset_new_len, test_dataset_len])
# test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)

## REMOVE
# change epochs back to 30
epochs = 10

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.AdamW(fcn_model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max =epochs, eta_min=1e-6)

fcn_model = fcn_model.to(device)



def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
        None.
    """

    best_iou_score = 0.0

    for epoch in range(epochs):
        ts = time.time()
        fcn_model.train()
        
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs =  inputs.to(device)
            labels =  labels.to(device)

            outputs = fcn_model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        scheduler.step()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score = val(epoch)

        ## REMOVE
        # break

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            torch.save(fcn_model.state_dict(), "best_model.pth")
    
def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_model.forward(inputs)
            output_labels = torch.argmax(outputs, dim=1)

            batch_loss = criterion(outputs, labels)
            batch_accuracy = util.pixel_acc(output_labels, labels)
            batch_iou = util.iou(output_labels, labels)
            
            mean_iou_scores.append(torch.mean(batch_iou).item())
            accuracy.append(torch.mean(batch_accuracy).item())
            losses.append(batch_loss.item())

    #print(mean_iou_scores[0:10])
            ## REMOVE
            # break


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

 #TODO
def modelTest():
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_model.forward(inputs)
            output_labels = torch.argmax(outputs, dim=1)

            batch_loss = criterion(outputs, labels)
            batch_accuracy = util.pixel_acc(output_labels, labels)
            batch_iou = util.iou(output_labels, labels)
            
            mean_iou_scores.append(torch.mean(batch_iou).item())
            accuracy.append(torch.mean(batch_accuracy).item())
            losses.append(batch_loss.item())


    print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    saved_model_path = "best_model.pth"
    ## REMOVE
    # TODO Then Load your best model using saved_model_path
    fcn_model.load_state_dict(torch.load(saved_model_path, map_location = device))
    
    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":

    val(0)  # show the accuracy before training

    ## REMOVE
    # UNCOMMENT
    train()
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
