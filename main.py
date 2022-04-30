from experiment import Experiment
import torch
# from hparam_tuning import Hyperparameter_Grid_Search
# ==============================================================================
# Import your models here:
from models.alexnet_cifar import Alexnet_cifar
from models.alexnet_cifar_gr import Alexnet_cifar_gr
from models.alexnet_cifar_gr_compare import Alexnet_cifar_gr_compare
from models.alexnet_cifar_gr_shuffle import Alexnet_cifar_gr_shuffle
from models.alexnet_flower import Alexnet_flower
from models.alexnet_flower_gr import Alexnet_flower_gr
from models.alexnet_flower_gr_compare import Alexnet_flower_gr_compare
from models.alexnet_flower_gr_shuffle import Alexnet_flower_gr_shuffle
from models.alexnet_miniImage import Alexnet_miniImage
from models.alexnet_miniImage_gr import Alexnet_miniImage_gr
from models.alexnet_miniImage_gr_shuffle import Alexnet_miniImage_gr_shuffle
from models.vgg16_cifar_gr_compare import vgg16_cifar_gr_compare
from models.vgg16_miniImage_gr_compare import vgg16_miniImage_gr_compare
from models.vgg16net_flower import vgg16net_flower
from models.vgg16net_flower_gr import vgg16_flowers_gr
from models.vgg16_cifar_gr import vgg16_cifar_gr
from models.vgg16net_cifar import vgg16net_cifar
from models.vgg16net_flower_gr_compare import vgg16_flowers_gr_compare
from models.vgg16net_miniImage import vgg16net_miniImage
from models.vgg16_miniImage_gr import vgg16_miniImage_gr
def main():
    torch.cuda.empty_cache()
    flower = (
        'daffodil', 'snowdrop', 'lilyValley', 'bluebell', 'crocys', 'iris', 
        'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'colts foot', 
        'dandelion', 'cowslip', 'buttercup', 'wind flower', 'pansy'
    )
    cifar=('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # ==========================================================================
    # Run an experiment
    # Please at least run the following parameters:
    
    # batch_size (int): 4, 64, 128
    # num_of_epoch (int): 100
    # optimizer (str): "Adam", "SGD"
    # learning_rate (float): 1e-4, 1e-3, 1e-2
    
    # model = ResNeXt()
    # model_name = "resneXt"
    # parameter = {
    #     "classes": classes,
    #     "model": model,
    #     "model_name": model_name,
    #     "batch_size": 64,
    #     "num_of_epoch": 100,
    #     "optimizer": "Adam",
    #     "learning_rate": 1e-4,
    # }
    # exp = Experiment(**parameter)
    # exp.train_val_evaluate()
    opt='Adam'
    lrr=2e-5

    model = Alexnet_miniImage(100)
    model_name = "Alexnet_miniImage_4"
    parameter = {
        "classes": cifar,
        "model": model,
        "model_name": model_name,
        "batch_size": 64,
        "num_of_epoch": 100,
        "optimizer": opt,
        "learning_rate": lrr,
    }
    exp = Experiment(**parameter)
    exp.train_val_evaluate()

    model = Alexnet_miniImage_gr(100)
    model_name = "Alexnet_miniImage_gr_4"
    parameter = {
        "classes": cifar,
        "model": model,
        "model_name": model_name,
        "batch_size": 64,
        "num_of_epoch": 100,
        "optimizer": opt,
        "learning_rate": lrr,
    }
    exp = Experiment(**parameter)
    exp.train_val_evaluate()

    # model = Alexnet_flower_gr_compare(17)
    # model_name = "Alexnet_flower_gr_compare"
    # parameter = {
    #     "classes": flower,
    #     "model": model,
    #     "model_name": model_name,
    #     "batch_size": 64,
    #     "num_of_epoch": 100,
    #     "optimizer": opt,
    #     "learning_rate": lrr,
    # }
    # exp = Experiment(**parameter)
    # exp.train_val_evaluate()

    # 

    # model =vgg16_cifar_gr_compare()
    # model_name = "vgg16_cifar_gr_compare"
    # parameter = {
    #     "classes": cifar,
    #     "model": model,
    #     "model_name": model_name,
    #     "batch_size": 64,
    #     "num_of_epoch": 50,
    #     "optimizer": opt,
    #     "learning_rate": lrr,
    # }
    # exp = Experiment(**parameter)
    # exp.train_val_evaluate()


    # model = vgg16_flowers_gr_compare()
    # model_name = "vgg16_flowers_gr_compare"
    # parameter = {
    #     "classes": flower,
    #     "model": model,
    #     "model_name": model_name,
    #     "batch_size": 32,
    #     "num_of_epoch": 100,
    #     "optimizer": opt,
    #     "learning_rate": lrr,
    # }
    # exp = Experiment(**parameter)
    # exp.train_val_evaluate()

    

    # model = vgg16_miniImage_gr_compare()
    # model_name = "vgg16_miniImage_gr_compare"
    # parameter = {
    #     "classes": flower,
    #     "model": model,
    #     "model_name": model_name,
    #     "batch_size": 64,
    #     "num_of_epoch": 100,
    #     "optimizer": opt,
    #     "learning_rate": lrr,
    # }
    # exp = Experiment(**parameter)
    # exp.train_val_evaluate()
    # ==========================================================================
    
if __name__ == '__main__':
    main()
