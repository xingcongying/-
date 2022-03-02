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

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')



opticalStreamNet=OpticalFlowStreamNet().to(device)

optimizer = optim.SGD(
    params=opticalStreamNet.parameters(),
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

    opticalStreamNet.train()
    for i in range(epoch):
        for index, data in enumerate(trainset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            
            optical_output=opticalStreamNet(OpticalFlow_images )
            optical_loss = F.cross_entropy(optical_output, label)
            

            optical_loss.backward()
         
            
            optimizer.step()

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('model/checkpoint-%i.pth' % iteration, opticalStreamNet, optimizer)     # OpticalFlow_ResNetModel

            iteration += 1
            print("optical_Loss: " + str(optical_loss.item()))
           
            with open('opticallog.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "s optical_Loss:"+str(optical_loss.item())+"\n")

        test(i+1)

    save_checkpoint('model/checkpoint-%i.pth' % iteration, opticalStreamNet, optimizer)


def test(i_epoch):

 
    opticalStreamNet.eval()

    optical_correct=0

    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)
            
            optical_output=opticalStreamNet(OpticalFlow_images )            
            optical_max_value,optical_max_index = optical_output.max(1, keepdim=True)
            optical_correct += optical_max_index.eq(label.view_as(optical_max_index)).sum().item()
            print("optical_max_index")
            print(optical_max_index)
            print("----------------------------------------------")
            print(label.view_as(optical_max_index))
            print(optical_correct)
            
           
          
          
    print("optical Accuracy: " + str(optical_correct*1.0*100/len(testset_loader.dataset)))
  
    with open('opticallog.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s optical Accuracy:"+str(optical_correct*1.0*100/len(testset_loader.dataset))+"\n")

if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)
    
