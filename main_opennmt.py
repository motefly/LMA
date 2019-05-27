import model_manager
import torch
import os
import datasets
import translation_models.model as tmm
import translation_models.help_fun as transl_hf
import onmt
import quantization
import pickle
import copy
import quantization.help_functions as qhf
import functools
import helpers.functions as mhf
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'LMA for Model Compression')
parser.add_argument('-batch_size', type = int, default=3192)
parser.add_argument('-test_batch_size', type = int, default=1024)
parser.add_argument('-init_lr', type = float, default=0.1)
parser.add_argument('-epochs', type = int, default=15)
parser.add_argument('-stud_act', type = str, default='relu')
parser.add_argument('-num_bins', type = int, default=8)
parser.add_argument('-plot_title', type=str, default='test')
parser.add_argument('-train_teacher', action='store_true')
parser.add_argument('-train_student', action='store_true')
parser.add_argument('-display_metrics', action='store_true')
parser.add_argument('-test_memory', action='store_true')
parser.add_argument('-seq_level', action='store_true')
parser.add_argument('-manager', type = str, default='model_manager_onmt')
parser.add_argument('-stModel', type = int, default=0)
parser.add_argument('-data', type = str, default='integ')
parser.add_argument('-model_name', type = str, default='')
parser.add_argument('-seed', type = int, default=1)

parser.add_argument('-n_layers', type = int, default=6)
parser.add_argument('-size', type = int, default=512)


args = parser.parse_args()

args.plot_title = 'summary/'+args.plot_title+'_seed_'+str(args.seed)

datasets.BASE_DATA_FOLDER = 'datas'
SAVED_MODELS_FOLDER = 'models'
MANAGER_FOLDER = 'manager'
USE_CUDA = torch.cuda.is_available()


torch.manual_seed(args.seed)
np.random.seed(args.seed)


#Define the architechtures we want to try
teacherOptions = copy.deepcopy(onmt.standard_options.stdOptions)
teacherOptions['epochs'] = args.epochs
teacherOptions['src_word_vec_size'] = args.size
teacherOptions['tgt_word_vec_size'] = args.size
teacherOptions['rnn_size'] = args.size
teacherOptions['layers'] = args.n_layers

smallerSizes = [256, 128, 64]
smallerLayers = [3, 3, 3]

smallerOptions = copy.deepcopy(onmt.standard_options.stdOptions)
smallerOptions['epochs'] = args.epochs
smallerOptions['src_word_vec_size'] = smallerSizes[args.stModel]
smallerOptions['tgt_word_vec_size'] = smallerSizes[args.stModel]
smallerOptions['rnn_size'] = smallerSizes[args.stModel]
smallerOptions['layers'] = smallerLayers[args.stModel]
smallerOptions['activation'] = args.stud_act
#smallerOptions['learning_rate'] = args.init_lr

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))
NUM_GPUS = len(cuda_devices)

if args.batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(args.batch_size, NUM_GPUS))

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass
try:
    os.mkdir(MANAGER_FOLDER)
except:pass

args.manager = args.data + '_' + args.manager

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

if args.data == 'integ':
    transl_dataset = datasets.onmt_integ_dataset(pin_memory=True)
elif args.data == 'wmt':
    transl_dataset = datasets.WMT13_DE_EN(pin_memory=True)

if args.test_memory:
    if args.train_teacher:
        modelOption = teacherOptions
    else:
        modelOption = smallerOptions
    model = tmm.create_model(transl_dataset.fields, options=modelOption)
        
    if USE_CUDA: model = model.cuda()
    translateOptions = onmt.standard_options.standardTranslationOptions
    translateOptions['batch_size'] = 1
    mem = transl_hf.translate_sequences(model, transl_dataset.processedFilesPath[0], modelOption,
                              translateOptions, transl_dataset.testFilesPath[0], test_memory = True)
    if args.train_teacher:
        str2save = 'teacher_{}: memory: {} M'.format(args.data, mem/(1024**2))
    else:
        str2save = 's_{}_{}_nb_{}_{}: memory: {} M'.format(args.stModel, args.stud_act, args.num_bins, args.data, mem/(1024**2))
    print(str2save)
    with open('memory.txt', 'a') as fr:
        fr.write(str2save+'\n')
    torch.cuda.empty_cache()
    import sys
    sys.exit()
    
train_loader, test_loader = transl_dataset.getTrainLoader(args.batch_size, 'tokens'), transl_dataset.getTestLoader(args.test_batch_size, 'tokens')

# Teacher model
model_name = args.manager+'_%s_teacher'%args.data

teacherModelPath = os.path.join(modelsFolder, model_name)
teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)

if USE_CUDA: teacherModel = teacherModel.cuda()
print('Parameters number of teacher model: '+str(sum(p.numel() for p in teacherModel.parameters())))
if NUM_GPUS > 1:
    teacherModel = torch.nn.parallel.DataParallel(teacherModel)

if not model_name in Manager.saved_models:
    Manager.add_new_model(model_name, teacherModelPath,
                          arguments_creator_function=teacherOptions)
if args.train_teacher:
    Manager.train_model(teacherModel, model_name=model_name,
                        train_function=tmm.train_model,
                        arguments_train_function={'options':teacherOptions,
                                                  'plot_path': args.plot_title+model_name},
                        train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(Manager.load_model_state_dict(model_name))
# cnn_hf.evaluateModel(teacherModel, test_loader, k=5)

#train normal distilled
model = tmm.create_model(transl_dataset.fields,
                         options=smallerOptions)

teacherModel_name = model_name
model_name = args.manager + '_%s_smaller_distilled_'%args.data + args.stud_act + '_s_' + str(args.stModel) + '_p_' + args.model_name+'_sed_'+str(args.seed)
model_path = os.path.join(modelsFolder, model_name)
if USE_CUDA: model = model.cuda()
print('Parameters number of student model: '+str(sum(p.numel() for p in model.parameters())))
if NUM_GPUS > 1:
    model = torch.nn.parallel.DataParallel(model)


standardTranslateOptions = onmt.standard_options.standardTranslationOptions
    
if args.train_student:
    folder_distillation_dataset = os.path.join(transl_dataset.dataFolder, 'distilled_dataset_' + teacherModel_name)
    if args.seq_level:
        try:
            distilled_dataset = datasets.translation_datasets.TranslationDataset(folder_distillation_dataset, src_language='de', tgt_language='en')#, pin_memory=True)
            trai_loader, test_loader = distilled_dataset.getTrainLoader(args.batch_size, 'tokens'), distilled_dataset.getTestLoader(args.test_batch_size, 'tokens')
            print('Distillation dataset loaded')
        except:
            print('Creating distillation dataset from scratch')
            transl_hf.create_distillation_dataset(teacherModel, teacherOptions, standardTranslateOptions, transl_dataset, folder_distillation_dataset)
            print('Distillation dataset created')
            distilled_dataset = datasets.translation_datasets.TranslationDataset(folder_distillation_dataset, src_language='de', tgt_language='en')# pin_memory=True)
            train_loader, test_loader = distilled_dataset.getTrainLoader(args.batch_size, 'tokens'), distilled_dataset.getTestLoader(args.test_batch_size, 'tokens')
            print('Distillation dataset loaded')
    Manager.remove_model(model_name)
# if not model_name in Manager.saved_models:
    Manager.add_new_model(model_name, model_path,
                          arguments_creator_function=smallerOptions)
    Manager.train_model(model, model_name=model_name,
                        train_function=tmm.train_model,
                        arguments_train_function={'options':smallerOptions,
                                                  'plot_path': args.plot_title+model_name,
                                                  'use_distillation_loss': True,
                                                  'teacher_model': teacherModel},
                        train_loader=train_loader, test_loader=test_loader)
# to complete
if args.display_metrics:
    file_results = 'results_file_BLEU_models.txt'
    display_models = []
    for x in Manager.list_models():
        if '{}_s_{}'.format(args.stud_act, args.stModel) in x:
            display_models.append(x)
        if 'teacher' in args.model_name and 'teacher' in x:
            display_models.append(x)
            break
    import pdb
    pdb.set_trace()
    for x in display_models:
        if Manager.get_num_training_runs(x) == 0:
            continue
        try:
            modelOptions = Manager.load_metadata(x, 0)[0]
        except:
            continue
        if 'activation' not in modelOptions:
            modelOptions['activation'] = 'relu'
        for key, val in modelOptions.items(): #remeding to an old bug in save_metadata function
            if val == 'None':
                modelOptions[key] = None
                
        if 'distilled' in x and 'word' not in x and args.seq_level:
            dataset = distilled_dataset
        else:
            dataset = transl_dataset
        model = tmm.create_model(dataset.fields, options=modelOptions)
        if USE_CUDA: model = model.cuda()
        try:
            model.load_state_dict(Manager.load_model_state_dict(x, 1))
        except:
            continue
    
        file_translation_model = transl_hf.get_translation_file_model(model, dataset, modelOptions, standardTranslateOptions,
                                                                      hypothesis_file_path='belu/{}.out'.format(x))
        bleu = transl_hf.get_bleu_moses(file_translation_model, dataset.testFilesPath[1], file_input=True)
        perplexity = Manager.load_metadata(x,1)[1]['perplexity'][-1]
        str_to_save = 'Model "{}"  ==> Perplexity: {}, BLEU: {}'.format(x,
                                                                        perplexity,
                                                                        bleu)
        with open(file_results, 'a') as fr:
            fr.write(str_to_save + '\n')
        print(str_to_save)
        
