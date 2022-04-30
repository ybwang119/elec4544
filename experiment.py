import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from MLclf import MLclf
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Import your dataset here
from custom_dataset import FlowerDataset

# TODO:
    # Classes order

class Experiment():
    def __init__(self, classes:tuple, model:nn.Module, model_name:str, 
        optimizer:str, learning_rate:float, num_of_epoch:int, batch_size:int,
        ) -> None:
        """ Initialize and Experiment

        Args:
            classes (tuple): A tuple of ordered string which are names of classes.
            model (nn.Module): The neural net to be trained.
            model_name (str): The name of the neural net. Used to store the neural net file
            optimizer (str): The name of the optimizer.
            learning_rate (float): The learning rate used for optimizer.
            num_of_epoch (int): Number of epochs.
            batch_size (int): The batch size.
        """
        
        # Store the label name
        self.classes = classes
        self.batch_size = batch_size
        # Load Data
        if 'flower' in model_name:
            img_dir = "flowers"
            train_dir = os.path.join(img_dir, "train")
            test_dir = os.path.join(img_dir, "test")
            val_dir = os.path.join(img_dir, "val")
            
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5],[.5,.5,.5])
            ])
            transform1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([256,256]),
                transforms.RandomCrop([224,224]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5],[.5,.5,.5])
            ])
            train_set = FlowerDataset(
                "labels.csv", 
                img_dir=train_dir, 
                transform=transform1
            )
            val_set = FlowerDataset(
                "labels.csv", 
                img_dir=val_dir, 
                transform=transform,
            )
            test_set = FlowerDataset(
                "labels.csv", 
                img_dir=test_dir, 
                transform=transform
            )
            
            train_loader = DataLoader(
                dataset = train_set,
                batch_size = batch_size,
                shuffle = True
            )
            val_loader = DataLoader(
                dataset = val_set,
                batch_size = batch_size,
                shuffle = False
            )
            test_loader = DataLoader(
                dataset = test_set,
                batch_size = batch_size,
                shuffle = False
            )
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
        elif 'cifar' in model_name:
            transform1=transforms.Compose([
                transforms.Resize([32,32]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5],[.5,.5,.5])   
            ])
            transform=transforms.Compose([
                transforms.Resize([32,32]),
                transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5],[.5,.5,.5])   
            ])
            train_set=torchvision.datasets.CIFAR10('./',True,transform1,download=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=0)
            val_set=torchvision.datasets.CIFAR10('./',False,transform,download=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=False, num_workers=0)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = val_loader
        elif 'miniImage' in model_name:
            transform1 = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Resize([84,84]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Resize([84,84]),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set, val_set, test_set = MLclf.miniimagenet_clf_dataset(ratio_train=0.8, ratio_val=0.1, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=0)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=False, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=0)
            self.train_loader=train_loader
            self.test_loader=test_loader
            self.val_loader=val_loader

        # CPU or CUDA
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Store the model used
        self.model = model.to(self.device)
        self.model_name = model_name
        
        # Store optimizer information
        # Here we only tune the learning rate, other parameters for opt algo are
        # the same as default value
        self.learning_rate = learning_rate
        optimizerD = {
            'Adam':optim.Adam(self.model.parameters(), lr=learning_rate), 
            'SGD':optim.SGD(self.model.parameters(), lr=learning_rate),
            'RMSprop':optim.RMSprop(self.model.parameters(), lr=learning_rate), 
            'Adadelta':optim.Adadelta(self.model.parameters(), lr=learning_rate),
            'Adagrad':optim.Adagrad(self.model.parameters(), lr=learning_rate), 
            'Adamax':optim.Adamax(self.model.parameters(), lr=learning_rate),
            'Nadam':optim.NAdam(self.model.parameters(), lr=learning_rate), 
        }
        self.optimizer_name = optimizer
        self.optimizer = optimizerD[optimizer]

        # Num of epoch in training
        self.num_of_epoch = num_of_epoch

        # For image classification task, we use cross entropy loss and do not
        # tune this paramter
        self.criterion = nn.CrossEntropyLoss()
        
        # Store the accuracy for validation set
        self.validation_accuracy = 0

        # Store the accuracy for test set
        self.test_accuracy = 0
        
        # Summary writer
        self.writer = SummaryWriter(comment=f"_{self.model_name}")


    def visualize_model_graph(self):
        dataiter = iter(self.train_loader)
        data = dataiter.next()
        images, labels = (
            data[0].to(self.device), data[1].to(self.device)
        )
        self.writer.add_graph(self.model, images)
        self.writer.flush()
    
    def train(self):
        for epoch in range(self.num_of_epoch):
            if epoch%30==0 and epoch!=0:
                self.learning_rate/=4
                self.optimizer=optim.Adam(self.model.parameters(), lr=self.learning_rate)
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                if i % 5 == 4: # print average loss of minibatch every 20 mini-batches
                    message = f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}"
                    print(message)

                    # Write to tensorboard
                    self.writer.add_scalar(
                        tag = "Training Loss",
                        scalar_value = running_loss / 4,
                        global_step = epoch * len(self.train_loader) + i,
                    )
                    # Reset running loss and running correct
                    running_loss = 0.0

            # Evaluate model on the validation set after each epoch 
            self.model.eval()
            (num_correct_pred_train, num_total_pred_train, _, class_predicted_train, 
                class_true_train) = self.evaluate(self.train_loader)
            fig_train = self.create_confusion_matrix(
                y_pred = class_predicted_train.data.cpu().numpy(), 
                y_true = class_true_train.data.cpu().numpy()
            ) 
            # self.writer.add_figure(
            #     tag = "Confusion Matrix Train (Step: Epoch)",
            #     figure = fig_train,
            #     global_step = epoch,
            # )
            self.writer.add_scalar(
                tag = "Training Accuracy (Step: Epoch)",
                scalar_value = num_correct_pred_train / num_total_pred_train,
                global_step = epoch,
            )
            (num_correct_pred_val, num_total_pred_val, _, class_predicted_val,
                class_true_val) = self.evaluate(self.val_loader)
            fig_val = self.create_confusion_matrix(
                y_pred = class_predicted_val.data.cpu().numpy(),
                y_true = class_true_val.data.cpu().numpy()
            )
            # self.writer.add_figure(
            #     tag = "Confusion Matrix Val (Step: Epoch)",
            #     figure = fig_val,
            #     global_step = epoch,
            # )
            self.writer.add_scalar(
                tag = "Validation Accuracy (Step: Epoch)",
                scalar_value = num_correct_pred_val / num_total_pred_val,
                global_step = epoch,
            )
            # self.writer.add_scalar(
            #     tag = "Validation top-5 Accuracy (Step: Epoch)",
            #     scalar_value = num_correct_top5_pred_val / num_total_pred_val,
            #     global_step = epoch,
            # )

            # Return to train mode
            self.model.train()

        # Save trained models
        PATH = os.path.join(
            "trained_models", 
            f'{self.model_name}_{int(time.time())}.pth'
        )
        torch.save(self.model.state_dict(), PATH)
    
    def images_to_probs(self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    
    def create_confusion_matrix(self, y_pred:list, y_true:list):
        # # Build confusion matrix
        # cf_matrix = confusion_matrix(y_true, y_pred)
        # df_cm = pd.DataFrame(
        #     cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
        #     index=[i for i in self.classes],
        #     columns=[i for i in self.classes]
        # )
        # plt.figure(figsize=(12, 7))    
        # return sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()
        pass
    def evaluate(self, data_loader, new_model=None):
        if new_model != None:
            model = new_model
        else:
            model = self.model
        
        # Evaluate performance in validaton/test set
        class_probs = []
        class_predicted = []
        class_label = []
        correct = 0
        #correct_5=0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                output = model(images)
                class_probs_batch = [F.softmax(el, dim=0) for el in output]
                _, predicted = torch.max(output.data, dim=1)
               # _, top5=torch.topk(output.data,5,dim=1)
               # label_5=labels.reshape(-1,1)
               # print('label_5')
               # print(label_5.shape)
               # temp=torch.ones((self.batch_size,5))
                #temp=temp.to(self.device)
               # label_5=temp*label_5
                correct += (predicted == labels).sum().item()
               # correct_5+=(label_5 == top5).sum().item()
                total += labels.size(0)

                class_predicted.append(predicted)
                class_probs.append(class_probs_batch)
                class_label.append(labels)
        class_predicted = torch.cat(class_predicted)
        class_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        class_label = torch.cat(class_label)
        
        return correct, total, class_probs, class_predicted, class_label

    def visualize_evaluation(self, correct, total, eval_probs, eval_label,test_flag=0):
        
        for i in range(len(self.classes)):
            tensorboard_truth = eval_label == i
            tensorboard_probs = eval_probs[:, i]
            self.writer.add_pr_curve(
                self.classes[i],
                tensorboard_truth,
                tensorboard_probs,
                global_step=0
            )
            self.writer.flush()
        
        if test_flag:
            accuracy_str = "test_accuracy"
        else:
            accuracy_str = "val_accuracy"
        self.writer.add_hparams(
            hparam_dict = {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "model_name": self.model_name,
                "optimizer": self.optimizer_name,
                "num_of_epoch": self.num_of_epoch,
            },
            metric_dict = {
                accuracy_str: correct / total,
            }
        )
        self.writer.flush()

    def train_val_evaluate(self):
        self.visualize_model_graph()
        self.train()
        self.model.eval()
        correct, total, class_probs, class_predicted, class_label = self.evaluate(
            data_loader = self.val_loader
        )
        self.visualize_evaluation(
            correct=correct, 
            total=total, 
            eval_probs=class_probs, 
            eval_label=class_label
        )