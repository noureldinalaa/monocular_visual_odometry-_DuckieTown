import torch.nn as nn
import torch.nn.functional as F
from Rotation_translation import Rotation_Translation
import pickle

import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image


Rt = Rotation_Translation()
# Rt.read_input_points()

with open("input_output_points.pickle","rb") as file:
    points = pickle.load(file)
input_points , output_points  = points[0],points[1]
print(input_points[0:2,:])
print()
print(output_points[0:2,:])


input_length , max_number,point_numbers = Rt.input_points_length()
env_points ,env_angles,env_length = Rt.env_data_points()
distance = Rt.env_distance_between_points(env_points ,env_length,point_numbers,input_length)
rotation = Rt.env_angles_between_points(env_angles, env_length,point_numbers,input_length)




input_points = input_points[:len(distance)]
print(len(input_points))

output_points = output_points[:len(distance)]
print(len(output_points))

x_input = input_points[:,0]
print(x_input[-1])
x_output = output_points[:,0]
print(x_output[-1])

y_input = input_points[:,1]
print(y_input[-1])
y_output = output_points[:,1]
print(y_output[-1])

x_input = list(map(float,x_input))
x_output = list(map(float,x_output))
x_difference = list(np.array(x_output) - np.array(x_input))
print(x_difference[-1])
print(len(x_difference))

y_input = list(map(float,y_input))
y_output = list(map(float,y_output))
y_difference = list(np.array(y_output) - np.array(y_input))
print(y_difference[-1])
print(len(y_difference))

final_input = list(zip(x_input,y_input,x_difference,y_difference))
print(final_input[-1])
final_output = list(zip(distance,rotation))
print(final_output[-1])

final_input = np.asarray(final_input,dtype=float)
print(final_input[-1])
final_output = np.asarray(final_output,dtype=float)
print(final_output[-1])


split_frac = 0.8
val_test_frac = 0.5
## split data into training, validation, and test data (features and labels, x and y)
split_idx = int(len(final_input)*split_frac)
train_x , remaining_x = final_input[:split_idx],final_input[split_idx:]
train_y , remaining_y = final_output[:split_idx],final_output[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x , test_x = remaining_x[:test_idx] , remaining_x[test_idx:]
val_y , test_y = remaining_y[:test_idx] , remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\t Feature shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set : \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
valid_data = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

# dataloaders
batch_size = 128

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Network()
# model = model.double()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

#Helper Function to calculate accuracy
# def accuracy(predection, y):
#     """ Return accuracy per batch. """
#     correct = (torch.round(torch.argmax(predection)) == y).float()
#     return correct.sum() / len(correct)

# def accuracy(predictions, labels):
#     classes = torch.argmax(predictions, dim=1)
#     return torch.mean((classes == labels).float())

# def calc_accuracy(prediction, Y):
#     """Calculates model accuracy
#
#     Arguments:
#         mdl {nn.model} -- nn model
#         X {torch.Tensor} -- input data
#         Y {torch.Tensor} -- labels/target values
#
#     Returns:
#         [torch.Tensor] -- accuracy
#     """
#     # _, predicted = torch.max(prediction, 1)
#     # total =+ Y.size(0)
#     # correct =+ (predicted == Y).sum().item()
#     # return  100 * correct / total
def calc_accuracy(pred,true):
    acc = (true.argmax(-1) == pred.argmax(-1)).float().cpu().detach().numpy()
    return float(100 * acc.sum() / len(acc))



# specify loss function
criterion = nn.MSELoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=0.00001)

# number of epochs to train the model
# n_epochs = 70000
# valid_loss_min = np.Inf  # track change in validation loss
# count = 0
# translation_rotation1 = torch.Tensor([])
# translation_rotation1 = translation_rotation1.cuda()
# translation_rotation2 = torch.Tensor([])
# translation_rotation2 = translation_rotation1.cuda()
# count_arr = []
#
# # to graph loses
# train_losses = []
# # train_losses = train_losses.cuda()
# valid_losses = []
# # valid_losses = valid_losses.cuda()
# for epoch in range(1, n_epochs + 1):
#
#     # keep track of training and validation loss
#     train_loss = 0.0
#     valid_loss = 0.0
#     train_acc = 0
#     valid_acc = 0
#     train_accuracy_epoch = 0.0
#
#     ###################
#     # train the model #
#     ###################
#     model.train()
#     for data, target in train_loader:
#         # move tensors to GPU if CUDA is available
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         # clear the gradients of all optimized variables
#         # print(data)
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)
#         # print(output)
#         # calculate the batch loss
#         loss = criterion(output, target)
#
#
#         # accuracy
#         train_accuracy_batch = calc_accuracy(output, target)
#         train_accuracy_epoch += train_accuracy_batch
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update training loss
#         # train_loss += loss.item() * data.size(0) * 100
#         train_loss += loss.item() * data.size(0)
#
#
#         # train_acc += train_accuracy.item() * data.size(0) * 1000
#
#         #         translation_rotation1.append(output)
#
#     #     translation_rotation1 = torch.cat([translation_rotation1, output], dim=0)
#     #     count_arr.append(count)
#     # #         print("this is length of output = " + str(len(output)))
#     # translation_rotation2 = torch.cat([translation_rotation2, translation_rotation1], dim=0)
#     # #     translation_rotation2.append(translation_rotation1)
#     # #     print("------->this is translation_rotation " + str(count) + " = " + str(translation_rotation2[count]))
#     # print("this is length of translation_rotation " + str(count) + " = " + str(len(translation_rotation2[count])))
#     # print(len(count_arr))
#     # count += 1
#
#     ######################
#     # validate the model #
#     ######################
#     model.eval()
#     for data, target in valid_loader:
#         # move tensors to GPU if CUDA is available
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)
#         # calculate the batch loss
#         loss = criterion(output, target)
#
#         # accuracy
#         # valid_accuracy = accuracy(output, target)
#
#         # update average validation loss
#         # valid_loss += loss.item() * data.size(0) * 100
#         valid_loss += loss.item() * data.size(0)
#         # valid_acc += valid_accuracy.item() * data.size(0) * 1000
#
#     # calculate average losses
#     train_loss = train_loss / len(train_loader.dataset)
#     valid_loss = valid_loss / len(valid_loader.dataset)
#
#     #lists to plot
#     train_losses = np.append(train_losses, train_loss)
#     valid_losses = np.append(valid_losses, valid_loss)
#
#     # calculate average acurracy
#     average_train_accuracy = train_accuracy_epoch / len(train_loader.dataset)
#     # average_valid_accuracy = valid_acc / len(train_loader.dataset)
#
#     # print training/validation statistics
#     print(
#         'Epoch: {} \tTraining Loss: {:.6f} \tTraining accuracy: {:.6f} \tValidation Loss: {:.6f} '.format(
#             epoch, train_loss,average_train_accuracy*100,valid_loss))
#
#     # save model if validation loss has decreased
#     if valid_loss <= valid_loss_min:
#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#             valid_loss_min,
#             valid_loss))
#         torch.save(model.state_dict(), 'model_trained.pt')
#         valid_loss_min = valid_loss



model.load_state_dict(torch.load('model_trained.pt'))

# plotting

# plt.plot(range(n_epochs),train_losses)
# plt.title("training_losses")
# plt.show(block = False)
# plt.savefig('train_losses.png')
# plt.pause(3)
# plt.close('all')
# plt.plot(range(n_epochs),valid_losses)
# plt.title("valid_losses")
# plt.show(block = False)
# plt.savefig('valid_losses.png')
# plt.pause(3)
# plt.close('all')


def test(loaders, model, criterion, train_on_gpu):
    # monitor test loss and accuracy
    test_loss = 0.
    correct_output = 0.
    correct_total = 0.0
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # # convert output probabilities to predicted class
        # pred = output.data.max(1, keepdim=True)[1]
        # # compare predictions to true label
        # correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        correct_batch = calc_accuracy(output,target)
        correct_total += correct_batch

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct_total / total, correct_total, total))


# call test function
# test(test_loader, model, criterion, train_on_gpu)
# Rt.optical_points_distance_histogram(x_difference,y_difference,input_length)
# Rt.env_dist_rotation_histogram(translation_rotation2,input_length)



def merged_plots_pred_vs_true(iteration_number_i):
    # length = len(input_length)
    # for iteration_number_i in range(length):
    im1 = Image.open('mixed_plots/self.id' + str(iteration_number_i) + '.png')
    im2 = Image.open('trained_distance_histograms/distance_trained_frame_'+str(iteration_number_i)+'.png')
    im3 = Image.open('trained_rotation_histograms/rotation_trained_frame_'+str(iteration_number_i)+'.png')

    imgs = [im1, im2,im3]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save('./merged_plots_pred_vs_true/merged_plots_' + str(iteration_number_i) + '.png')

# merged_plots_pred_vs_true()

def predicted_dist_rotation_histogram():
    features = final_input[:]
    true_labels = final_output[:]
    features = torch.from_numpy(features).float()
    i = 0
    loop = 0
    var_skip = 0
    model.eval()
    for data in features:
        if train_on_gpu:
            data =data.cuda()

        output = model(data)
        predicted_distance = output [0]
        predicted_rotation = output [1]

        predicted_distance = predicted_distance.cpu().detach().numpy()
        predicted_rotation = predicted_rotation.cpu().detach().numpy()

        if loop == var_skip:
            true_distance = true_labels[loop][0]
            plt.hist(predicted_distance, bins=35,rwidth=0.4,label='predicted_distance='+str(predicted_distance))
            plt.hist(true_distance, bins=35,rwidth=0.2 ,label='true_distance='+str(true_distance))
            plt.legend(loc='upper right',prop={'size': 6})
            plt.title("env_distance_trained_frame_"+str(i))
            plt.savefig('./trained_distance_histograms/' + 'distance_trained_frame_'+str(i)+'.png')
            #plt.show(block = False)
            plt.close('all')


            true_rotation = true_labels[loop][1]
            plt.hist(predicted_rotation, bins=35 ,rwidth=0.4 ,label='predicted_rotation='+str(predicted_rotation))
            plt.hist(true_rotation, bins=35,rwidth=0.2 , label='true_rotation='+str(true_rotation))
            plt.legend(loc='upper right',prop={'size': 6})
            plt.title("env_rotation_trained_frame_"+str(i))
            plt.savefig('./trained_rotation_histograms/' + 'rotation_trained_frame_'+str(i)+'.png')
            #plt.show(block = False)
            plt.close('all')

            merged_plots_pred_vs_true(i)
            i +=1
            # print(i)
            if i!=len(input_length):
                var_skip += int(input_length[i])

        loop +=1

predicted_dist_rotation_histogram()

