import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from data_reader import data_reader, data_reader_with_normalization, read_val_data_with_normalization, write_labeled, write_submision
from torchvision.models import resnet50, ResNet50_Weights

cuda0 = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.manual_seed(13)

def train(trainloader, validloader, net, path, epochs = 10, l_rate = 0.001): 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9)

    training_loss=[]
    validation_loss=[]
    training_acc = []
    validation_acc = []

    minim_val_loss = 100
    for epoch in range(epochs):

        net.train()

        running_loss = 0.0
        count = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(cuda0, dtype=torch.float32), labels.to(cuda0, dtype=torch.long)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        training_loss.append(running_loss / count)
        acc = compute_acc(net, trainloader)
        training_acc.append(acc)
        print(f'Epoch {epoch + 1} -> training_loss: {running_loss / count} training_acc: {acc}')

        net.eval()
        running_loss = 0.0
        count = 0
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(cuda0, dtype=torch.float32), labels.to(cuda0, dtype=torch.long)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            count += 1

        validation_loss.append(running_loss / count)
        acc = compute_acc(net, validloader)
        validation_acc.append(acc)
        print(f'Epoch {epoch + 1} -> validation_loss: {running_loss / count} validation_acc: {acc}')

        if running_loss / count < minim_val_loss:
          torch.save(net.state_dict(), path)
          minim_val_loss = running_loss / count
          print(f'Saving model')

    print('Finished Training')
    
    return training_loss, validation_loss, training_acc, validation_acc

def compute_acc(net, data_loader):
    correct = 0
    total = 0
    all_labeles = []
    all_predicted = []
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(cuda0, dtype=torch.float32), labels.to(cuda0, dtype=torch.long)
            outputs = net(images).cuda()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labeles = all_labeles + labels.cpu().tolist()
            all_predicted = all_predicted + predicted.cpu().tolist()

    return correct / total

def predict(net, test_Data):
    all_predicted = []
    net.eval()
    with torch.no_grad():
        for image in test_Data:
            image =  torch.reshape(image, (1, 3, 64, 64)) 
            outputs = net(image.to(cuda0, dtype=torch.float32)).cuda()
            predicted = torch.max(outputs.data, 1)
            all_predicted.append(predicted.indices.cpu().item())
        

    return all_predicted

def predict_loader(net, data_loader):
    all_predicted = []
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            images = data.to(cuda0, dtype=torch.float32)
            outputs = net(images).cuda()
            _, predicted = torch.max(outputs.data, 1)
            all_predicted = all_predicted + predicted.cpu().tolist()
            
    return all_predicted

def predict_loader_confidence(net, data_loader):
    all_predicted = []
    index = 0
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            images = data.to(cuda0, dtype=torch.float32)
            outputs = net(images).cuda()
            predicted_softmax = torch.nn.functional.softmax(outputs, 1)

            values, predicted = torch.max(predicted_softmax.data, 1)
            predicted = predicted.cpu().tolist()
            values = values.cpu().tolist()
            for i in range(len(predicted)):
                if values[i] > 0.9:
                    all_predicted.append((index, predicted[i]))
                index = index + 1
            
    return all_predicted

def predict_unlabeled(model, dataset_path = './task1/train_data/images/unlabeled/', result_path = 'predicted_unlabeled.csv'):
    # Predict unlabeled data
    test_loader = read_val_data_with_normalization(dataset_path, 26445, 64)
    all_predicted = predict_loader_confidence(model, test_loader)
    write_labeled(all_predicted, result_path)

def task_1(path_dataset = './task1', model_path = './models/model_lr-1e-3-normalization-v15.pth', submission_path = './sub_task1/submission15.csv', batch_size = 64, lr = 1e-3, epochs = 10):
    train_loader, val_loader = data_reader_with_normalization(path_dataset + "/train_data/annotations.csv", batch_size, enhance=True)
    model_resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    num_ftrs = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_ftrs, 100)
    model_convd = model_resnet50.to(cuda0)
    
    train(train_loader, val_loader, model_convd, model_path, epochs, lr)

    model_convd.load_state_dict(torch.load(model_path))
    model_convd.eval()

    test_loader = read_val_data_with_normalization(path_dataset + '/val_data', 5000, batch_size)
    all_predicted = predict_loader(model_resnet50, test_loader)
    write_submision(all_predicted, submission_path)

def task_2(path_dataset = './task2', model_path = './models_task2/model_lr-1e-3-normalization-v3.pth', submission_path = './sub_task2/submission3.csv', batch_size = 64, lr = 1e-3, epochs = 10):
    train_loader, val_loader = data_reader_with_normalization(path_dataset + "/train_data/annotations.csv", batch_size, 40000)
    model_resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    num_ftrs = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_ftrs, 100)
    model_convd = model_resnet50.to(cuda0)

    train(train_loader, val_loader, model_convd, model_path, epochs, lr)

    model_convd.load_state_dict(torch.load(model_path))
    model_convd.eval()

    test_loader = read_val_data_with_normalization(path_dataset + '/val_data/', 5000, batch_size)
    all_predicted = predict_loader(model_resnet50, test_loader)
    write_submision(all_predicted, submission_path)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--task', type=str)
    arg_parser.add_argument('--dataset', type=str)
    arg_parser.add_argument('--model_path', type=str)
    arg_parser.add_argument('--result_path', type=str)
    args = arg_parser.parse_args()
    
    if args.task == '1':
        task_1(args.dataset, args.model_path, args.result_path)
    
    if args.task == '2':
        task_2(args.dataset, args.model_path, args.result_path)

