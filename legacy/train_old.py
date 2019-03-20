# # Import
# from __future__ import division
# from vision_utils.custom_torch_utils import create_summary_writer
# from vision_utils.custom_torch_utils import create_supervised_trainer_multitask, create_supervised_evaluator_multitask
# from vision_utils.custom_torch_utils import my_multi_task_loss, MutliTaskLoss, MultiTaskAccuracy
# from multitask_rag.utk_data_utils import get_utk_dataloader
# import argparse
# from tqdm import tqdm
# from ignite.engine.engine import Events
# from ignite import handlers
# from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
# import torch
# import os
# import glob
# import shutil
# import numpy as np
#
#
# PATH_TO_MODEL_SCRIPT = './model_configs/sep_conv_adam.py'
# # SRC_DIR = "/media/sf_Documents/COMPUTER_VISION/UTKface_Aligned_cropped/UTKFace"
# # DEST_DIR = "/media/sf_Documents/COMPUTER_VISION/UTKface_Aligned_cropped/utk_face_split"
# # TRAIN_SPLIT = 0.7
# SRC_DIR = None
# DEST_DIR = None
# TRAIN_SPLIT = None
# CHECKPOINT = './checkpoint'
# FILE_NAME = 'sep_conv_adam'
# LOG_INTERVAL = 2
# EPOCHS = 2
#
# # variable to track validation loss and computing it separately for each handler (checkpoint, early stop, ...)
# val_loss = [np.inf]
#
#
# def run_utk(path_to_model_script, epochs, log_interval, dataloaders,
#             dirname='resnet_models', filename_prefix='resnet', n_saved=2,
#             log_dir='../../fer2013/logs', launch_tensorboard=False, patience=10,
#             resume_model=None, resume_optimizer=None, backup_step=1, backup_path=None,
#             n_epochs_freeze=5, n_cycle=None, lr_after_freeze=1e-3, lr_cycle_start=1e-4, lr_cycle_end=1e-1,
#             loss_weights=[1/10, 1/0.16, 1/0.44]):
#
#     """
#
#     :param path_to_model_script:
#     :param epochs:
#     :param log_interval:
#     :param dataloaders:
#     :param dirname:
#     :param filename_prefix:
#     :param n_saved:
#     :param log_dir:
#     :param launch_tensorboard:
#     :param patience:
#     :param resume_model:
#     :param resume_optimizer:
#     :param backup_step:
#     :param backup_path:
#     :return:
#     """
#
#     if launch_tensorboard:
#         os.makedirs(log_dir, exist_ok=True)
#     #     os.system('pkill tensorboard')
#     #     os.system('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(log_dir))
#     #     os.system("npm install -g localtunnel")
#     #     os.system('lt --port 6006 >> /content/url.txt 2>&1 &')
#     #     os.system('cat /content/url.txt')
#
#     # Get the model, optimizer and dataloaders from script
#     model_script = dict()
#     with open(path_to_model_script) as f:
#         exec(f.read(), model_script)
#
#     model = model_script['my_model']
#     optimizer = model_script['optimizer']
#
#     # continue with a trained model
#     if resume_model:
#         model.load_state_dict(torch.load(resume_model))
#
#     if resume_optimizer:
#         optimizer.load_state_dict(torch.load(resume_optimizer))
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if torch.is_tensor(v):
#                     state[k] = v.cuda()
#
#     train_loader, val_loader = dataloaders['train'], dataloaders['valid']
#
#     if launch_tensorboard:
#         writer, val_writer = create_summary_writer(model, train_loader, log_dir)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     trainer = create_supervised_trainer_multitask(model, optimizer, loss_fn=my_multi_task_loss,
#                                                   loss_weights=loss_weights, device=device)
#     evaluator = create_supervised_evaluator_multitask(model,
#                                                       metrics={'mt_accuracy': MultiTaskAccuracy(),
#                                                                'mt_loss': MutliTaskLoss()},
#                                                       device=device, loss_weights=loss_weights)
#
#     # function to schedule learning rate if needed
#     @trainer.on(Events.EPOCH_STARTED)
#     def schedule_learning_rate(engine):
#         if engine.state.epoch > n_epochs_freeze and n_cycle not in [None, 0] \
#                 and not getattr(trainer, 'scheduler_set', False):
#             scheduler = LinearCyclicalScheduler(optimizer, 'lr', lr_cycle_start,
#                                                 lr_cycle_end, len(train_loader) * n_cycle)
#             trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
#             setattr(trainer, 'scheduler_set', True)
#
#     desc = "ITERATION - loss: {:.3f}"
#     pbar = tqdm(
#         initial=0, leave=False, total=len(train_loader),
#         desc=desc.format(0)
#     )
#
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def log_training_loss(engine):
#         iter_ = (engine.state.iteration - 1) % len(train_loader) + 1
#
#         if iter_ % log_interval == 0:
#             pbar.desc = desc.format(engine.state.output)
#             pbar.update(log_interval)
#
#         if launch_tensorboard:
#             writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_training_results(engine):
#         pbar.refresh()
#         evaluator.run(train_loader)
#         metrics = evaluator.state.metrics
#         age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
#         avg_nll = metrics['mt_loss']
#         tqdm.write(
#             "Training Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} "
#             "** Race accuracy: {:.3f} ** Avg loss: {:.3f}"
#             .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll)
#         )
#
#         if launch_tensorboard:
#             writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
#             writer.add_scalar('age_l1_loss', age_l1_loss, engine.state.epoch)
#             writer.add_scalar('gender_accuracy', gender_acc, engine.state.epoch)
#             writer.add_scalar('race_accuracy', race_acc, engine.state.epoch)
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_validation_results(engine):
#         evaluator.run(val_loader)
#         metrics = evaluator.state.metrics
#         age_l1_loss, gender_acc, race_acc = metrics['mt_accuracy']
#         avg_nll = metrics['mt_loss']
#         tqdm.write(
#             "Validation Results - Epoch: {} Age L1-loss: {:.3f} ** Gender accuracy: {:.3f} **"
#             " Race accuracy: {:.3f} ** Avg loss: {:.3f}"
#             .format(engine.state.epoch, age_l1_loss, gender_acc, race_acc, avg_nll))
#
#         pbar.n = pbar.last_print_n = 0
#
#         global val_loss
#         val_loss.append(avg_nll)
#
#         if launch_tensorboard:
#             val_writer.add_scalar('avg_loss', avg_nll, engine.state.epoch)
#             val_writer.add_scalar('age_l1_loss', age_l1_loss, engine.state.epoch)
#             val_writer.add_scalar('gender_accuracy', gender_acc, engine.state.epoch)
#             val_writer.add_scalar('race_accuracy', race_acc, engine.state.epoch)
#
#     @trainer.on(Events.EPOCH_STARTED)
#     def unfreeze(engine):
#         if engine.state.epoch == n_epochs_freeze:
#             for param in model.parameters():
#                 if not param.requires_grad:
#                     param.requires_grad = True
#                     optimizer.add_param_group(
#                         {'params': param, "lr": lr_after_freeze}
#                     )
#
#     def get_val_loss(engine):
#         global val_loss
#         return -val_loss[-1]
#
#     checkpointer = handlers.ModelCheckpoint(dirname=dirname, filename_prefix=filename_prefix,
#                                             score_function=get_val_loss,
#                                             # score_function=log_validation_results,
#                                             score_name='val_loss',
#                                             n_saved=n_saved, create_dir=True,
#                                             require_empty=False, save_as_state_dict=True
#                                             )
#     earlystop = handlers.EarlyStopping(patience, get_val_loss, trainer)
#     #
#     evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
#                                 {'optimizer': optimizer, 'model': model})
#     evaluator.add_event_handler(Events.EPOCH_COMPLETED, earlystop)
#
#     # optimizer and model that are in the gdrive, created from a previous run
#     original_files = glob.glob(os.path.join(backup_path, '*.pth*'))
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def backup_checkpoints(engine):
#         if backup_path is not None:
#             if engine.state.epoch % backup_step == 0:
#                 # get old model and optimizer files paths so that we can remove them after copying the newer ones
#                 old_files = glob.glob(os.path.join(backup_path, '*.pth'))
#
#                 # get new model and optimizer checkpoints
#                 new_files = glob.glob(os.path.join(dirname, '*.pth*'))
#                 if len(new_files) > 0:  # copy new checkpoints from local checkpoint folder to the backup_path folder
#                     for f_ in new_files:
#                         shutil.copy2(f_, backup_path)
#
#                     if len(old_files) > 0:  # remove older checkpoints as the new ones have been copied
#                         for f_ in old_files:
#                             if f_ not in original_files:
#                                 os.remove(f_)
#
#     if launch_tensorboard:
#         @trainer.on(Events.EPOCH_COMPLETED)
#         def add_histograms(engine):
#             for name, param in model.named_parameters():
#                 writer.add_histogram(name, param.clone().cpu().data.numpy(), engine.state.epoch)
#
#     trainer.run(train_loader, max_epochs=epochs)
#     pbar.close()
#     if launch_tensorboard:
#         writer.close()
#         val_writer.close()
#
#
# def main(args=None):
#
#     if args is None:
#         parser = argparse.ArgumentParser('Train a pytorch model')
#         parser.add_argument('--resize', type=int, default=128,
#                             help='int representing height and width for resizing the input image')
#
#         parser.add_argument('--normalize', type=int, default=0,
#                             help='whether to normalize (1) or not (0), useful for imagenet pretrained models')
#
#         parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for '
#                                                                                'train and validation data')
#         parser.add_argument('--n_samples', type=int, default=None, help='Number of images to sample,'
#                                                                         ' useful for debugging with small sets')
#         parser.add_argument('--path_to_model_script', type=str, default=PATH_TO_MODEL_SCRIPT,
#                             help='path to the script containing model and optimizer definition')
#         parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training iterations')
#         parser.add_argument('--data_dir', type=str, default=DATA_DIR,
#                             help='root directory containing train, valid and test image folders')
#         parser.add_argument('--src_dir', type=str, default=SRC_DIR, help='source directory containing '
#                                                                          'raw images if they are '
#                                                                          'not already split in train-test-valid')
#         parser.add_argument('--dest_dir', type=str, default=DEST_DIR, help='destination where to store train, val '
#                                                                            'and test sub-folders after split')
#         parser.add_argument('--train_split', type=float, default=None,
#                             help='proportion of train split, in the range 0 to 1')
#         parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT,
#                             help='folder to save checkpoints')
#         parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL,
#                             help='Print metrics each log_interval iterations')
#         parser.add_argument('--file_name', type=str, default=FILE_NAME,
#                             help='filename under which to save the checkpoints')
#         parser.add_argument('--n_saved', type=int, default=2, help='Save the n_saved best models')
#         parser.add_argument('--log_dir', type=str, default='./', help='directory where to save tensorboard logs')
#         parser.add_argument('--patience', type=int, default=10,
#                             help='Patience in terms of number of epochs for early stopping')
#         parser.add_argument('--launch_tensorboard', type=int, default=0,
#                             help='whether to start tensorboard automatically (0) or not (1)')
#         parser.add_argument('--resume_model', type=str, default=None,
#                             help='if given, path to an old model checkpoint from which to restore weights')
#         parser.add_argument('--resume_optimizer', type=str, default=None,
#                             help='if given, path to an old optimizer checkpoint from which to restore state from a '
#                                  'previous run')
#         parser.add_argument('--backup_step', type=int, default=1,
#                             help='backup current checkpoints in a given directory every backup_step epochs')
#         parser.add_argument('--backup_path', type=str, default=None,
#                             help='path to folder where to backup current checkpoints, typically when training on'
#                                  'google colab this is a path to a folder in my google drive '
#                                  'so that I can periodically copy my model checkpoints to google drive')
#         parser.add_argument('--n_epochs_freeze', type=int, default=5,
#                          help='number of epochs after which to unfreeze the model parameters, useful for finetuning')
#         parser.add_argument('--n_cycle', type=int, default=None,
#                             help='number of epochs for which to complete a learning rate scheduling cycle')
#         parser.add_argument('--lr_after_freeze', type=int, default=1e-3,
#                             help='set new learning rate after unfreezing layers')
#         parser.add_argument('--lr_cycle_start', type=int, default=1e-4,
#                             help='start value for learning rate in case of cyclical scheduling i.e. `n_cycle`>0')
#         parser.add_argument('--lr_cycle_end', type=int, default=1e-1,
#                             help='end value for learning rate in case of cyclical scheduling i.e. `n_cycle`>0')
#
#         args = parser.parse_args()
#
#     int2bool = {0: False, 1: True}
#
#     print('-----------Creating data loaders---------------------')
#     resize = None
#     if args.resize is not None:
#         resize = (args.resize,) * 2
#
#     dataloaders = get_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir,
#                                   resize=resize,
#                                   normalize=int2bool[args.normalize],
#                                   n_samples=args.n_samples,
#                                   src_dir=args.src_dir,
#                                   dest_dir=args.dest_dir,
#                                   train_split=args.train_split
#                                   )
#
#     print('-------------------- start training--------------------')
#     run_utk(args.path_to_model_script, epochs=args.epochs, log_interval=args.log_interval,
#             dataloaders=dataloaders, dirname=args.checkpoint_dir, filename_prefix=args.file_name,
#             n_saved=args.n_saved, log_dir=args.log_dir,
#             launch_tensorboard=int2bool[args.launch_tensorboard], patience=args.patience,
#             resume_model=args.resume_model, resume_optimizer=args.resume_optimizer,
#             backup_step=args.backup_step, backup_path=args.backup_path,
#             n_epochs_freeze=args.n_epochs_freeze, n_cycle=args.n_cycle,
#             lr_after_freeze=args.lr_after_freeze, lr_cycle_start=args.lr_cycle_start, lr_cycle_end=args.lr_cycle_end)
#
#
# if __name__ == '__main__':
#
#     main()
