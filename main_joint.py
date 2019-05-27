import model_manager
import torch
import os
import datasets
import cnn_models.conv_forward_model as convForwModel
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
parser.add_argument('-batch_size', type = int, default=64)
parser.add_argument('-init_lr', type = float, default=1e-2)
parser.add_argument('-epochs', type = int, default=200)
parser.add_argument('-stud_act', type = str, default='relu')
parser.add_argument('-num_bins', type = int, default=8)
parser.add_argument('-plot_title', type=str, default='test')
parser.add_argument('-train_teacher', action='store_true')
parser.add_argument('-train_student', action='store_true')
parser.add_argument('-manager', type = str, default='model_manager_cifar10')
parser.add_argument('-stModel', type = int, default=0)
parser.add_argument('-data', type = str, default='cifar10')
parser.add_argument('-seed', type = int, default=1)
parser.add_argument('-numBits', type = str, default='8')

args = parser.parse_args()

args.plot_title = 'summary/'+args.plot_title+'_seed_'+str(args.seed)
args.numBits = [int(x) for x in args.numBits.split(',')]

datasets.BASE_DATA_FOLDER = 'datas'
SAVED_MODELS_FOLDER = 'models'
MANAGER_FOLDER = 'manager'
USE_CUDA = torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Define the architechtures we want to try
smallerModelSpec0 = {'spec_conv_layers': [(75, 5, 5), (50, 5, 5), (50, 5, 5), (25, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [500], 'width': 32, 'height': 32}
smallerModelSpec1 = {'spec_conv_layers': [(50, 5, 5), (25, 5, 5), (25, 5, 5), (10, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [400], 'width': 32, 'height': 32}
smallerModelSpec2 = {'spec_conv_layers': [(25, 5, 5), (10, 5, 5), (10, 5, 5), (5, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [300], 'width': 32, 'height': 32}

smallerModelSpecs = [smallerModelSpec0, smallerModelSpec1, smallerModelSpec2]

print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

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

if args.data == 'cifar10':
    data = datasets.CIFAR10()
elif args.data == 'cifar100':
    data = datasets.CIFAR100()
train_loader, test_loader = data.getTrainLoader(args.batch_size), data.getTestLoader(args.batch_size)

# Teacher model
model_name = args.manager+'_%s_teacher'%args.data
teacherModelPath = os.path.join(modelsFolder, model_name)
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
# teacherModel = convForwModel.ConvolForwardNet(**smallerModelSpec0,
                                              # activation=args.act,
                                              useBatchNorm=USE_BATCH_NORM,
                                              useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
if USE_CUDA: teacherModel = teacherModel.cuda()
print('Parameters number of teacher model: '+str(sum(p.numel() for p in teacherModel.parameters())))

if not model_name in Manager.saved_models:
    Manager.add_new_model(model_name, teacherModelPath,
                                 arguments_creator_function={**convForwModel.teacherModelSpec,
                                 # arguments_creator_function={**smallerModelSpec0,
                                                             # 'activation':args.act,
                                                             'useBatchNorm':USE_BATCH_NORM,
                                                             'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
if args.train_teacher:
    Manager.train_model(teacherModel, model_name=model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                         'initial_learning_rate': args.init_lr,
                                                         'plot_path': args.plot_title+model_name},
                               train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(Manager.load_model_state_dict(model_name))
# cnn_hf.evaluateModel(teacherModel, test_loader, k=5)


#train normal distilled

model_name = args.manager + '_%s_smaller_distilled_'%args.data + args.stud_act + '_s_' + str(args.stModel) + 'tmp'

if args.train_student:
    for numBit in args.numBits:
        model_name = model_name + '_distil_quantized{}bits'.format(numBit)
        quantized_model_path = os.path.join(modelsFolder, model_name)
        quantized_model = convForwModel.ConvolForwardNet(**smallerModelSpecs[args.stModel],
                                                         activation=args.stud_act,
                                                         numBins=args.num_bins,
                                                         useBatchNorm=USE_BATCH_NORM,
                                                         useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
        if USE_CUDA: quantized_model = quantized_model.cuda()
        print('Parameters number of student model: '+str(sum(p.numel() for p in quantized_model.parameters())))
        Manager.remove_model(model_name)
        
        Manager.add_new_model(model_name, quantized_model_path,
                              arguments_creator_function={**smallerModelSpecs[args.stModel],
                                                          'activation':args.stud_act,
                                                          'numBins':args.num_bins,
                                                          'useBatchNorm':USE_BATCH_NORM,
                                                          'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
        
        Manager.train_model(quantized_model, model_name=model_name,
                            train_function=convForwModel.train_model,
                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                      'quantizeWeights':True,
                                                      'numBits':numBit,
                                                      'bucket_size':256,
                                                      'use_distillation_loss': True,
                                                      'plot_path': args.plot_title+model_name,
                                                      'teacher_model': teacherModel,
                                                      'quantize_first_and_last_layer':False},
                            train_loader=train_loader, test_loader=test_loader)
