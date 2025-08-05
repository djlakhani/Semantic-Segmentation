from transfer_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from torchvision.utils import save_image

num_workers = multiprocessing.cpu_count()

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if not hasattr(m, 'weight') or m.weight is None:
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

def getClassWeights(num_classes, dataloader):
    class_counts = torch.zeros(num_classes)

    for i, (inputs, labels) in enumerate(dataloader):
        classes, counts = torch.unique(labels, return_counts=True)
        class_counts[classes] += counts

    class_counts = torch.clamp(class_counts, min=1e-6)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    return class_weights

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

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)

epochs = 20

n_class = 21

fcn_transfer_model = FCNTransfer(n_class=n_class)
fcn_transfer_model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_weights = getClassWeights(n_class, train_loader)
class_weights = class_weights.to(device)

optimizer = torch.optim.AdamW(fcn_transfer_model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =epochs, eta_min=1e-6)

fcn_transfer_model = fcn_transfer_model.to(device)



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

    best_accuracy_score = 0.0
    train_loss = []
    val_loss = []
    epochs_count = []
    best_model_epoch = 0
    train_loss_avg = 0.0

    for epoch in range(epochs):
        ts = time.time()
        fcn_transfer_model.train()
        num_batches = 0
        
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs =  inputs.to(device)
            labels =  labels.to(device)

            outputs = fcn_transfer_model.forward(inputs)
            loss = criterion(outputs, labels)
            train_loss_avg = train_loss_avg + loss.item()
            num_batches = num_batches + 1

            loss.backward()
            optimizer.step()

            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        # for plotting purposes
        train_loss_avg = train_loss_avg / num_batches
        train_loss.append(train_loss_avg)
        train_loss_avg = 0.0
        num_batches = 0
        epochs_count.append(epoch)
        
        scheduler.step()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_accuracy, current_miou_score, current_val_loss = val(epoch)
        val_loss.append(current_val_loss)

        if current_accuracy > best_accuracy_score:
            best_accuracy_score = current_accuracy
            best_model_epoch = epoch
            torch.save(fcn_transfer_model.state_dict(), "best_transfer_model.pth")

    return train_loss, val_loss, epochs_count, best_model_epoch
    
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
    fcn_transfer_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_transfer_model.forward(inputs)
            output_labels = torch.argmax(outputs, dim=1)

            batch_loss = criterion(outputs, labels)
            batch_accuracy = util.pixel_acc(output_labels, labels)
            batch_iou = util.iou(output_labels, labels)
            
            mean_iou_scores.append(torch.mean(batch_iou).item())
            accuracy.append(torch.mean(batch_accuracy).item())
            losses.append(batch_loss.item())


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_transfer_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(accuracy), np.mean(mean_iou_scores), np.mean(losses)


def convert_class_color(output):
    """
    Maps each class in the ouput prediction to a color.
    Colors determined by palette.
    """
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
    
    dicti = {}
    c = 0
    for i in range(0,len(palette),3):
        dicti[c] = palette[i:i+3]
        c+=1
    
    palette = dicti

    output_color_mapped = torch.zeros([224, 224, 3])

    for i in range(224):
        for j in range(224):
            pixel_class = output[i][j]
            rgb = palette[pixel_class.item()]
            color_ten = torch.tensor([rgb[0], rgb[1], rgb[2]])
            output_color_mapped[i][j] = color_ten

    return output_color_mapped

def mask_overlay(input_img, pred_mask):
    """
    Generates an image with the predicted, translucent mask overlaying the image.
    """
    # print(input_img.shape)
    # print(pred_mask.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow((input_img.permute(1, 2, 0)).cpu().numpy())
    plt.imshow(pred_mask.cpu().numpy(), alpha=0.5)
    plt.savefig('output_mask_img_overlay.png')


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

    fcn_transfer_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    saved_model_path = "best_transfer_model.pth"
    fcn_transfer_model.load_state_dict(torch.load(saved_model_path, map_location = device))

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_transfer_model.forward(inputs)
            output_labels = torch.argmax(outputs, dim=1)

            batch_loss = criterion(outputs, labels)
            batch_accuracy = util.pixel_acc(output_labels, labels)
            batch_iou = util.iou(output_labels, labels)
            
            mean_iou_scores.append(torch.mean(batch_iou).item())
            accuracy.append(torch.mean(batch_accuracy).item())
            losses.append(batch_loss.item())

            # plotting samples from test dataset
            class_convert_sample = convert_class_color(output_labels[10])
            labels_convert_sample = convert_class_color(labels[10])
            save_image(labels_convert_sample.permute(2, 0, 1), 'true_mask_train.png')
            save_image(class_convert_sample.permute(2, 0, 1), 'image_mask_train.png')
            save_image(inputs[10], 'image_input_train.png')
            mask_overlay(inputs[10], class_convert_sample)
            break


    print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")

    fcn_transfer_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


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

    fcn_transfer_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    saved_model_path = "best_transfer_model.pth"
    fcn_transfer_model.load_state_dict(torch.load(saved_model_path, map_location = device))
    
    inputs = inputs.to(device)
    
    output_image = fcn_transfer_model(inputs)
    
    fcn_transfer_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

def plot_model(train_loss, val_loss, epoch_count, best_model_epoch):
    fig, ax = plt.subplots()
    ax.plot(epoch_count, train_loss, marker='o', linestyle='-', label='Train Loss', color='red')
    ax.plot(epoch_count, val_loss, marker='o', linestyle='-', label='Validation Loss', color='green')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss over Epochs')
    ax.legend()
    ax.set_xticks(epoch_count)
    plt.axvline(x=best_model_epoch, color='blue', linestyle='--', label='Best model')
    plt.savefig('loss_transfer_plt.png')
    plt.show()
    

if __name__ == "__main__":

    # val(0)  # show the accuracy before training
    # train_loss, val_loss, epoch_count, best_model_epoch = train()
    # plot_model(train_loss, val_loss, epoch_count, best_model_epoch)
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
