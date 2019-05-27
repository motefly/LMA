import model_manager
import torch
import os
import datasets
import cnn_models.conv_forward_model as convForwModel
from cnn_models.wide_resnet import Wide_ResNet
import cnn_models.resnet_kfilters as resnet_kfilters
import cnn_models.help_fun as cnn_hf
import quantization
import pickle
import copy
import quantization.help_functions as qhf
import functools
import helpers.functions as mhf
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'LMA for Model Compression')
parser.add_argument('-batch_size', type = int, default=128)
parser.add_argument('-init_lr', type = float, default=0.1)
parser.add_argument('-epochs', type = int, default=200)
parser.add_argument('-stud_act', type = str, default='relu')
parser.add_argument('-num_bins', type = int, default=8)
parser.add_argument('-plot_title', type=str, default='test')
parser.add_argument('-train_teacher', action='store_true')
parser.add_argument('-train_student', action='store_true')
parser.add_argument('-test_memory', action='store_true')
parser.add_argument('-manager', type = str, default='model_manager_cifar100')
parser.add_argument('-stModel', type = int, default=0)
parser.add_argument('-data', type = str, default='cifar100')
parser.add_argument('-model_name', type = str, default='')
parser.add_argument('-seed', type = int, default=1)

args = parser.parse_args()

args.plot_title = 'summary/'+args.plot_title+'_seed_'+str(args.seed)

datasets.BASE_DATA_FOLDER = 'datas'
SAVED_MODELS_FOLDER = 'models'
MANAGER_FOLDER = 'manager'
USE_CUDA = torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Define the architechtures we want to try

# teacherOptions = {'widen_factor':12, 'depth':28, 'dropout_rate':0.3, 'num_classes':100}

smallerModelSpec0 = {'widen_factor':4, 'depth':10, 'dropout_rate':0.3, 'num_classes':100}
smallerModelSpec1 = {'widen_factor':2, 'depth':10, 'dropout_rate':0.3, 'num_classes':100}

teacherOptions = {'widen_factor':10, 'depth':16, 'dropout_rate':0.3, 'num_classes':100}
smallerModelSpecs = [smallerModelSpec0, smallerModelSpec1]

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))
NUM_GPUS = len(cuda_devices)
# add new, check for lma on multi-gpu performance
args.batch_size = args.batch_size * NUM_GPUS

if args.batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass
try:
    os.mkdir(MANAGER_FOLDER)
except:pass

manager_path = os.path.join(MANAGER_FOLDER, args.manager+'.tst')
create_new = True
if os.path.exists(manager_path):
    create_new = False
Manager = model_manager.ModelManager(manager_path,
                                     args.manager,
                                     create_new_model_manager=create_new)
modelsFolder = os.path.join(SAVED_MODELS_FOLDER, args.data)

# for x in Manager.list_models():
#     if Manager.get_num_training_runs(x) >= 1:
#         print(x, Manager.load_metadata(x)[1]['predictionAccuracy'][-1])

try:
    os.mkdir(modelsFolder)
except:pass

epochsToTrainCIFAR = args.epochs
USE_BATCH_NORM = True
AFFINE_BATCH_NORM = True

TRAIN_SMALLER_MODEL = False
TRAIN_SMALLER_QUANTIZED_MODEL = False
TRAIN_DISTILLED_MODEL = False
TRAIN_DIFFERENTIABLE_QUANTIZATION = False
CHECK_PM_QUANTIZATION = True

if args.data == 'cifar10':
    data = datasets.CIFAR10()
elif args.data == 'cifar100':
    data = datasets.CIFAR100()

if args.test_memory:
    if not args.train_teacher:
        model = Wide_ResNet(**smallerModelSpecs[args.stModel],
                    activation=args.stud_act,
                    numBins=args.num_bins)
    else:
        model = Wide_ResNet(**teacherOptions)
        
    if USE_CUDA: model = model.cuda()
    test_loader = data.getTestLoader(1)
    import time
    start = time.time()
    cnn_hf.evaluateModel(model, test_loader)
    mem = torch.cuda.max_memory_allocated()
    end = time.time()
    avg_time = (end-start)*1000 / len(test_loader)
    if args.train_teacher:
        str2save = 'teacher_cifar100: time: {} ms, memory: {} M'.format(avg_time, mem/(1024**2))
        print(str2save)
    else:
        str2save = 's_{}_{}_nb_{}cifar100: time: {} ms, memory: {} M'.format(args.stModel, args.stud_act, args.num_bins, avg_time, mem/(1024**2))
        print(str2save)
    with open('memory.txt', 'a') as fr:
        fr.write(str2save+'\n')
    torch.cuda.empty_cache()
    import sys
    sys.exit()

train_loader, test_loader = data.getTrainLoader(args.batch_size), data.getTestLoader(args.batch_size)

# Teacher model
model_name = args.manager+'_%s_teacher'%args.data

teacherModelPath = os.path.join(modelsFolder, model_name)
teacherModel = Wide_ResNet(**teacherOptions)
# teacherModel = resnet_kfilters.resnet18(num_classes=100)

if USE_CUDA: teacherModel = teacherModel.cuda()
print('Parameters number of teacher model: '+str(sum(p.numel() for p in teacherModel.parameters())))
if NUM_GPUS > 1:
    teacherModel = torch.nn.parallel.DataParallel(teacherModel)

if not model_name in Manager.saved_models:
    Manager.add_new_model(model_name, teacherModelPath,
                          arguments_creator_function=teacherOptions)
if args.train_teacher:
    Manager.train_model(teacherModel, model_name=model_name,
                        train_function=convForwModel.train_model,
                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                  'initial_learning_rate': args.init_lr,
                                                  'learning_rate_style': 'cifar100',
                                                  'print_every':50,
                                                  'weight_decayL2': 0.0005,
                                                  'plot_path': args.plot_title+model_name},
                        train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(Manager.load_model_state_dict(model_name))
# cnn_hf.evaluateModel(teacherModel, test_loader, k=5)

#train normal distilled
model = Wide_ResNet(**smallerModelSpecs[args.stModel],
                    activation=args.stud_act,
                    numBins=args.num_bins)
                                       
model_name = args.manager + '_%s_smaller_distilled_'%args.data + args.stud_act + '_s_' + str(args.stModel) + '_p_' + args.model_name
model_path = os.path.join(modelsFolder, model_name)
if USE_CUDA: model = model.cuda()
print('Parameters number of student model: '+str(sum(p.numel() for p in model.parameters())))
if NUM_GPUS > 1:
    model = torch.nn.parallel.DataParallel(model)

if args.train_student:
    Manager.remove_model(model_name)
# if not model_name in Manager.saved_models:
    Manager.add_new_model(model_name, model_path,
                          arguments_creator_function={**smallerModelSpecs[args.stModel],
                                                      'activation':args.stud_act,
                                                      'numBins':args.num_bins})
    Manager.train_model(model, model_name=model_name,
                        train_function=convForwModel.train_model,
                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                  'initial_learning_rate': args.init_lr,
                                                  'learning_rate_style': 'cifar100',#'quant_points_cifar100', #'cifar100',
                                                  'activation':args.stud_act,
                                                  'print_every':50,
                                                  'weight_decayL2': 0.0005,
                                                  'use_distillation_loss': True,
                                                  'plot_path': args.plot_title+model_name,
                                                  'teacher_model': teacherModel},
                                                  # 'ask_teacher_strategy':('incorrect_labels',None)},
                        train_loader=train_loader, test_loader=test_loader)

