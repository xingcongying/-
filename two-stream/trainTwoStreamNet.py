from LoadUCF101Data import trainset_loader, testset_loader
from Two_Stream_Net import TwoStreamNet,OpticalFlowStreamNet,RGBStreamNet
import torch
import torch.optim as optim
import torch.nn.functional as F



EPOCH = 100
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
SAVE_INTERVAL = 500

import os
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')


    
twoStreamNet = TwoStreamNet().to(device)
opticalFlowStreamNet=OpticalFlowStreamNet().to(device)
rgbStreamNet=RGBStreamNet().to(device)

rgb_optimizer = optim.SGD(
    params=twoStreamNet.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)

opticalFlow_optimizer = optim.SGD(
    params=twoStreamNet.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)

optimizer = optim.SGD(
    params=twoStreamNet.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

    

def train(epoch, save_interval):
    iteration = 0
    twoStreamNet.train()
    opticalFlowStreamNet.train()
    rgbStreamNet.train()
    for i in range(epoch):
        for index, data in enumerate(trainset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            rgb_optimizer.zero_grad()
            opticalFlowStreamNet.zero_grad()
            
            rgb_output=rgbStreamNet(RGB_images )
            rgb_loss = F.cross_entropy(rgb_output, label)
            
            opticalFlow_output=opticalFlowStreamNet(OpticalFlow_images)
            opticalFlow_loss = F.cross_entropy(opticalFlow_output, label)
            
            output = twoStreamNet(RGB_images, OpticalFlow_images)
            loss = F.cross_entropy(output, label)
            
            rgb_loss.backward()
            opticalFlow_loss.backward()
            loss.backward()

            rgb_optimizer.step()
            optimizer.step()
            opticalFlow_optimizer.step()
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('model/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)     # OpticalFlow_ResNetModel

            iteration += 1
            print("rgb_Loss: " + str(rgb_loss.item()))
            print("opticalFlow_Loss: " + str(opticalFlow_loss.item()))
            print("Loss: " + str(loss.item()))
            with open('log.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()) + ",rgb_Loss:"+str(rgb_loss.item())+",opticalFlow_Loss:"+str(opticalFlow_loss.item())+"\n")

        test(i+1)

    save_checkpoint('model/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)


def test(i_epoch):

    twoStreamNet.eval()
    opticalFlowStreamNet.eval()
    rgbStreamNet.eval()
    correct = 0
    rgb_correct=0
    opticalFlow_correct=0
    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)
            
            rgb_output=rgbStreamNet(RGB_images )
            opticalFlow_output=opticalFlowStreamNet(OpticalFlow_images)
            output = twoStreamNet(RGB_images, OpticalFlow_images)

            
            rgb_max_value, rgb_max_index = rgb_output.max(1, keepdim=True)
            rgb_correct += rgb_max_index.eq(label.view_as(rgb_max_index)).sum().item()
            print("rgb_max_index")
            print(rgb_max_index)
            
            print("----------------------------------------------")
            opticalFlow_max_value, opticalFlow_max_index = opticalFlow_output.max(1, keepdim=True)
            opticalFlow_correct +=opticalFlow_max_index.eq(label.view_as(opticalFlow_max_index)).sum().item()
            print("opticalFlow_max_index")
            print(opticalFlow_max_index)
            
            print("----------------------------------------------")
          
            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()
            print("max_index")
            print(max_index)
            print("label.view_as(max_index)")
            print(label.view_as(max_index))
    print("rgb Accuracy: " + str(rgb_correct*1.0*100/len(testset_loader.dataset)))
    print("opticalFlow Accuracy: " + str(opticalFlow_correct*1.0*100/len(testset_loader.dataset)))
    print("Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)))
    with open('log.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)) + ",rgb Accuracy:"+str(rgb_correct*1.0*100/len(testset_loader.dataset))+",opticalFlow Accuracy:" +str(opticalFlow_correct*1.0*100/len(testset_loader.dataset))+"\n")

if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)
    
