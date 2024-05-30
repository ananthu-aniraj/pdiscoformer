import copy
import os
from dataclasses import asdict
from typing import Dict, List, Any

import fsspec
import numpy as np
from timm.data import Mixup
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
from utils.training_utils.snapshot_class import Snapshot
from utils.data_utils.reversible_affine_transform import generate_affine_trans_params
from utils.data_utils.class_balanced_distributed_sampler import ClassBalancedDistributedSampler
from utils.data_utils.class_balanced_sampler import ClassBalancedRandomSampler
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.visualize_att_maps import VisualizeAttentionMaps
from utils.training_utils.engine_utils import load_state_dict_pdisco, AverageMeter
from utils.wandb_params import init_wandb
from .losses import *


class PDiscoTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss_fn: List[torch.nn.Module],
            save_every: int,
            snapshot_path: str,
            loggers: List,
            log_freq: int = 10,
            use_amp: bool = False,
            grad_norm_clip: float = 1.0,
            max_epochs: int = 100,
            num_workers: int = 4,
            mixup_fn: Optional[Mixup] = None,
            eval_only: bool = False,
            loss_hyperparams: Optional[Dict] = None,
            eq_affine_transform_params: Optional[Dict] = None,
            use_ddp: bool = True,
            sub_path_test: str = "",
            dataset_name: str = "",
            amap_saving_prob: float = 0.05,
            class_balanced_sampling: bool = False,
            num_samples_per_class: int = 100,
    ) -> None:
        self._init_ddp(use_ddp)
        self.num_landmarks = model.num_landmarks
        self.num_classes = model.num_classes
        # Top-k accuracy metrics for evaluation
        self._init_accuracy_metrics()
        self.model = model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.name_dataset = dataset_name
        self.sub_path_test = sub_path_test
        self.batch_size = batch_size
        self.eval_only = eval_only
        # Number of samples per class for class balanced sampling
        self.num_samples_per_class = num_samples_per_class
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers,
                                                     class_balanced_sampling=class_balanced_sampling)
        self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
        if len(loss_fn) == 1:
            self.loss_fn_train = self.loss_fn_eval = loss_fn[0]
        else:
            self.loss_fn_train = loss_fn[0]
            self.loss_fn_eval = loss_fn[1]

        self.save_every = save_every
        self.amap_saving_prob = amap_saving_prob
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.isdir(snapshot_path):
            self.is_snapshot_dir = True
        else:
            self.is_snapshot_dir = False
        if loggers:
            if self.local_rank == 0 and self.global_rank == 0:
                loggers[0] = init_wandb(loggers[0])
        self.loggers = loggers
        self.log_freq = log_freq
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.mixup_fn = mixup_fn
        self.epoch_test_accuracies = []
        self.current_epoch = 0
        self.accum_steps = 1

        # Equivariance affine transform parameters
        self._init_affine_transform_params(eq_affine_transform_params)

        # Loss hyperparameters
        self._init_losses(loss_hyperparams)

        # Loss dictionary
        self._init_loss_dict()

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if os.path.isfile(os.path.join(snapshot_path, f"snapshot_best.pt")):
            print("Loading snapshot")
            self._load_snapshot()
        elif os.path.isfile(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()
            self.snapshot_path = os.path.dirname(snapshot_path)
            self.is_snapshot_dir = True
        self.batch_img_metas = None
        # Initialize the visualization class
        self.vis_att_maps = VisualizeAttentionMaps(snapshot_dir=self.snapshot_path, sub_path_test=self.sub_path_test,
                                                   dataset_name=self.name_dataset, bg_label=self.num_landmarks,
                                                   batch_size=self.batch_size, num_parts=self.num_landmarks + 1)
        if self.use_ddp:
            if self.local_rank == 0 and self.global_rank == 0:
                print(f"Using {self.world_size} GPUs, Broadcast Buffers")
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=True)
        else:
            print("Using single GPU")

        self.epoch_test_accuracies = []
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(model, log="all", log_freq=self.log_freq)

    def _init_ddp(self, use_ddp) -> None:
        self.is_slurm_job = "SLURM_NODEID" in os.environ
        self.use_ddp = use_ddp
        if self.is_slurm_job:
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.global_rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_world_size = self.world_size // n_nodes
            self.use_ddp = True
        else:
            if not self.use_ddp:
                self.local_rank = 0
                self.global_rank = 0
                self.world_size = 1
                self.local_world_size = 1
            else:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    def _init_losses(self, loss_hyperparams: dict) -> None:
        # Loss hyperparameters
        self.l_classification = loss_hyperparams['l_class_att']
        self.l_presence = loss_hyperparams['l_presence']
        self.l_presence_beta = loss_hyperparams['l_presence_beta']
        self.l_presence_type = loss_hyperparams['l_presence_type']
        self.l_conc = loss_hyperparams['l_conc']
        self.l_orth = loss_hyperparams['l_orth']
        self.l_equiv = loss_hyperparams['l_equiv']
        self.l_tv = loss_hyperparams['l_tv']
        self.l_enforced_presence = loss_hyperparams['l_enforced_presence']
        self.l_enforced_presence_loss_type = loss_hyperparams['l_enforced_presence_loss_type']
        self.l_pixel_wise_entropy = loss_hyperparams['l_pixel_wise_entropy']
        self.conc_loss = ConcentrationLoss().to(self.local_rank, non_blocking=True)
        self.enforced_presence_loss = EnforcedPresenceLoss(loss_type=self.l_enforced_presence_loss_type).to(
            self.local_rank,
            non_blocking=True)
        self.total_variation_loss = TotalVariationLoss(reduction="mean").to(self.local_rank, non_blocking=True)
        self.presence_loss = PresenceLoss(beta=self.l_presence_beta,
                                          loss_type=self.l_presence_type).to(self.local_rank, non_blocking=True)
        self.loss_fn_eval = self.loss_fn_eval.to(self.local_rank, non_blocking=True)
        self.loss_fn_train = self.loss_fn_train.to(self.local_rank, non_blocking=True)

    def _init_affine_transform_params(self, eq_affine_transform_params: dict) -> None:
        # Equivariance affine transform parameters
        self.eq_degrees = eq_affine_transform_params['degrees']
        self.eq_translate = eq_affine_transform_params['translate']
        self.eq_scale_ranges = eq_affine_transform_params['scale_ranges']
        self.eq_shear = eq_affine_transform_params['shear']

    def _init_loss_dict(self) -> None:
        self.loss_dict_train = {'loss_classification_train': AverageMeter(),
                                'loss_conc_train': AverageMeter(),
                                'loss_presence_train': AverageMeter(),
                                'loss_orth_train': AverageMeter(),
                                'loss_equiv_train': AverageMeter(),
                                'loss_total_train': AverageMeter(),
                                'loss_tv': AverageMeter(),
                                'loss_enforced_presence': AverageMeter(),
                                'loss_pixel_wise_entropy': AverageMeter()}

        self.loss_dict_val = {'loss_total_val': AverageMeter()}

    def _init_accuracy_metrics(self) -> None:
        self.acc_dict_train = {'train_acc': torchmetrics.classification.MulticlassAccuracy(
                                num_classes=self.num_classes, top_k=1,
                                average="micro").to(self.local_rank,
                                                    non_blocking=True),
                               'train_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="micro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top1_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=1,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top5_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True)}

        self.acc_dict_test = {'test_acc': torchmetrics.classification.MulticlassAccuracy(
                                num_classes=self.num_classes, top_k=1,
                                average="micro").to(self.local_rank,
                                                    non_blocking=True),
                              'test_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="micro").to(self.local_rank,
                                                       non_blocking=True),
                              'macro_avg_acc_top1_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=1,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True),
                              'macro_avg_acc_top5_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=5,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True)}

    def _prepare_dataloader_ddp(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                                class_balanced_sampling: bool = False):
        if class_balanced_sampling:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                sampler=ClassBalancedDistributedSampler(dataset, num_samples_per_class=self.num_samples_per_class)
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                sampler=DistributedSampler(dataset)
            )

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                            class_balanced_sampling: bool = False, drop_last: bool = True):
        if self.use_ddp:
            return self._prepare_dataloader_ddp(dataset, num_workers, class_balanced_sampling)

        if class_balanced_sampling:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                sampler=ClassBalancedRandomSampler(dataset, num_samples_per_class=self.num_samples_per_class))
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                drop_last=drop_last,
            )

    def _load_snapshot(self) -> None:
        loc = f"cuda:{self.local_rank}"
        try:
            if self.is_snapshot_dir:
                snapshot = fsspec.open(os.path.join(self.snapshot_path, f"snapshot_best.pt"))
            else:
                snapshot = fsspec.open(self.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location=loc)
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot, state_dict = load_state_dict_pdisco(snapshot_data)
        self.model.load_state_dict(state_dict)
        if self.eval_only:
            return
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.scheduler.step(snapshot.finished_epoch)
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True, vis_att_maps: bool = False, curr_iter: int = 0) -> \
            Tuple[Any, Any]:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.use_amp):
            all_features, maps, scores, dis_sim_maps = self.model(source)

            outputs = scores.mean(dim=-1)  # (batch_size, num_classes)

            if train:
                # Forward pass of transformed images
                angle, translate, scale, shear = generate_affine_trans_params(
                    degrees=self.eq_degrees, translate=self.eq_translate, scale_ranges=self.eq_scale_ranges,
                    shears=self.eq_shear, img_size=[source.shape[2], source.shape[3]])
                # Apply the affine transform to the source image
                source_transformed = rigid_transform(img=source, angle=angle,
                                                     translate=translate,
                                                     scale=scale,
                                                     shear=0.0, invert=False)
                equiv_maps = self.model(source_transformed)[1]

                # Classification loss
                loss_classification = self.loss_fn_train(outputs, targets) * self.l_classification

                # Concentration loss
                if self.l_conc > 0:
                    loss_conc = self.conc_loss(maps) * self.l_conc
                else:
                    loss_conc = torch.tensor(0.0, device=self.local_rank)

                # Total variation loss
                loss_tv = self.total_variation_loss(maps) * self.l_tv

                # Presence loss (fg) for landmarks
                loss_presence = self.presence_loss(maps=maps[:, :-1, :, :]) * self.l_presence

                # Orthogonality loss
                loss_orth = orthogonality_loss(all_features) * self.l_orth

                # Equivariance loss: calculate rotated landmarks distance
                loss_equiv = equivariance_loss(maps, equiv_maps, source, self.num_landmarks, translate, angle, scale,
                                               shear=0.0) * self.l_equiv

                # Enforced presence loss
                loss_enforced_presence = self.enforced_presence_loss(maps) * self.l_enforced_presence

                # Pixel-wise entropy loss
                loss_pixel_wise_entropy = pixel_wise_entropy_loss(maps) * self.l_pixel_wise_entropy

                loss = loss_conc + loss_presence + loss_classification + loss_orth + loss_equiv + loss_tv + loss_enforced_presence + loss_pixel_wise_entropy

                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.optimizer.step()

                losses_dict = {'loss_classification_train': loss_classification.item(),
                               'loss_conc_train': loss_conc.item(),
                               'loss_presence_train': loss_presence.item(),
                               'loss_orth_train': loss_orth.item(),
                               'loss_equiv_train': loss_equiv.item(),
                               'loss_total_train': loss.item(), 'loss_tv': loss_tv.item(),
                               'loss_enforced_presence': loss_enforced_presence.item(),
                               'loss_pixel_wise_entropy': loss_pixel_wise_entropy.item()}
            else:
                loss = self.loss_fn_eval(outputs, targets)
                losses_dict = {'loss_total_val': loss.item()}
                if vis_att_maps:
                    if np.random.random() < self.amap_saving_prob:
                        self.vis_att_maps.show_maps(ims=source, maps=maps, epoch=self.current_epoch,
                                                    curr_iter=curr_iter)
        return outputs, losses_dict

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        """
        Runs one epoch of training or evaluation
        :param epoch: Current epoch
        :param dataloader: Dataloader to use
        :param train: If we are training or evaluating
        :return:
        loss: Average loss across all batches
        top1: Average top1 accuracy across all batches
        top5: Average top5 accuracy across all batches
        losses_dict: Dictionary of all losses
        """

        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)

        last_accum_steps = len(dataloader) % self.accum_steps
        updates_per_epoch = (len(dataloader) + self.accum_steps - 1) // self.accum_steps
        num_updates = (epoch - 1) * updates_per_epoch
        last_batch_idx = len(dataloader) - 1
        last_batch_idx_to_accum = len(dataloader) - last_accum_steps

        vis_att_maps = True if epoch % self.save_every == 0 else False
        vis_att_maps = True if epoch == self.max_epochs else vis_att_maps
        vis_att_maps = True if epoch == 1 else vis_att_maps
        losses_dict = {}
        for key in self.loss_dict_train.keys():
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val.keys():
            self.loss_dict_val[key].reset()

        accuracies_dict = {}

        for key in self.acc_dict_train.keys():
            self.acc_dict_train[key].reset()
        for key in self.acc_dict_test.keys():
            self.acc_dict_test[key].reset()

        for it, mini_batch in enumerate(dataloader):
            source = mini_batch[0]
            targets = mini_batch[1]
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)

            batch_preds, losses_dict = self._run_batch(source, targets, train,
                                                       vis_att_maps=vis_att_maps, curr_iter=it)

            if train:
                for key in losses_dict.keys():
                    self.loss_dict_train[key].update(losses_dict[key], source.size(0))
                if self.mixup_fn is None:
                    for key in self.acc_dict_train.keys():
                        self.acc_dict_train[key].update(batch_preds, targets)
                num_updates += 1
                self.scheduler.step_update(num_updates=num_updates)
            else:
                for key in losses_dict.keys():
                    self.loss_dict_val[key].update(losses_dict[key], source.size(0))
                for key in self.acc_dict_test.keys():
                    self.acc_dict_test[key].update(batch_preds, targets)
            if it % self.log_freq == 0:
                if train:
                    print(
                        f'[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Total Loss {losses_dict["loss_total_train"]:.5f}'
                        f'| Classification Loss {losses_dict["loss_classification_train"]:.5f} | Concentration Loss {losses_dict["loss_conc_train"]:.5f}'
                        f'| Presence Loss {losses_dict["loss_presence_train"]:.5f} | Orth Loss {losses_dict["loss_orth_train"]:.5f}'
                        f'| Equiv Loss {losses_dict["loss_equiv_train"]:.5f} | TV Loss {losses_dict["loss_tv"]:.5f}'
                        f'| Enforced Presence Loss {losses_dict["loss_enforced_presence"]:.5f}'
                        f'| Pixel-wise Entropy Loss {losses_dict["loss_pixel_wise_entropy"]:.5f}')
                else:
                    print(
                        f'[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} '
                        f'Total Loss {losses_dict["loss_total_val"]:.5f}')
        if train:
            for key in self.loss_dict_train.keys():
                losses_dict[key] = self.loss_dict_train[key].avg
            if self.mixup_fn is None:
                for key in self.acc_dict_train.keys():
                    accuracies_dict[key] = self.acc_dict_train[key].compute().item() * 100
            self.scheduler.step(epoch)
        else:
            for key in self.loss_dict_val.keys():
                losses_dict[key] = self.loss_dict_val[key].avg
            for key in self.acc_dict_test.keys():
                accuracies_dict[key] = self.acc_dict_test[key].compute().item() * 100
        return losses_dict, accuracies_dict

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            epoch_test_accuracies=self.epoch_test_accuracies,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        if self.is_snapshot_dir:
            save_path_base = self.snapshot_path
        else:
            save_path_base = os.path.dirname(self.snapshot_path)
        if epoch == self.max_epochs:
            save_path = os.path.join(save_path_base, f"snapshot_final.pt")
        elif save_best:
            save_path = os.path.join(save_path_base, f"snapshot_best.pt")
        else:
            save_path = os.path.join(save_path_base, f"snapshot_{epoch}.pt")

        torch.save(snapshot, save_path)
        print(f"Snapshot saved at epoch {epoch}")

    def finish_logging(self):
        for logger in self.loggers:
            logger.finish()

    def train(self):
        for epoch in range(self.epochs_run, self.max_epochs):
            epoch += 1
            self.current_epoch = epoch
            self.model.train()
            loss_dict_train, acc_dict_train = self._run_epoch(epoch, self.train_loader, train=True)

            logging_dict = {'epoch': epoch,
                            'base_lr': self.optimizer.param_groups[0]['lr'],
                            'scratch_lr': self.optimizer.param_groups[-1]['lr'],
                            'modulation_lr': self.optimizer.param_groups[-2]['lr'],
                            'finer_lr': self.optimizer.param_groups[-3]['lr']}
            if self.local_rank == 0 and self.global_rank == 0:
                logging_dict.update(loss_dict_train)
                logging_dict.update(acc_dict_train)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self.model.eval()
                loss_dict_val, acc_dict_test = self._run_epoch(epoch, self.test_loader, train=False)
                if self.local_rank == 0 and self.global_rank == 0:
                    test_acc = acc_dict_test['test_acc']
                    self.epoch_test_accuracies.append(test_acc)
                    max_acc = max(self.epoch_test_accuracies)
                    max_acc_index = self.epoch_test_accuracies.index(max_acc)
                    if max_acc_index == len(self.epoch_test_accuracies) - 1:
                        self._save_snapshot(epoch, save_best=True)

                    logging_dict.update(loss_dict_val)
                    logging_dict.update(acc_dict_test)
                    for logger in self.loggers:
                        logger.log(logging_dict)

        if self.local_rank == 0 and self.global_rank == 0:
            self.finish_logging()

    def test_only(self):
        self.model.eval()
        logging_dict = {}
        with torch.inference_mode():
            if self.test_loader:
                loss_dict_val, acc_dict_test = self._run_epoch(0, self.test_loader, train=False)
            print(
                f'Test loss: {loss_dict_val["loss_total_val"]:.5f} '
                f'| Test acc: {acc_dict_test["test_acc"]:.5f} '
                f'| Test acc top5: {acc_dict_test["test_acc_top5"]:.5f} '
                f'| Macro avg acc top1: {acc_dict_test["macro_avg_acc_top1_test"]:.5f} '
                f'| Macro avg acc top5: {acc_dict_test["macro_avg_acc_top5_test"]:.5f}')

        if self.local_rank == 0 and self.global_rank == 0:
            logging_dict.update(loss_dict_val)
            logging_dict.update({'epoch': 0})
            logging_dict.update(acc_dict_test)
            for logger in self.loggers:
                logger.log(logging_dict)
        self.finish_logging()


def launch_pdisco_trainer(model: torch.nn.Module,
                          train_dataset: torch.utils.data.Dataset,
                          test_dataset: torch.utils.data.Dataset,
                          batch_size: int,
                          optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler.LRScheduler,
                          loss_fn: List[torch.nn.Module],
                          epochs: int,
                          save_every: int,
                          loggers: List,
                          log_freq: int,
                          use_amp: bool = False,
                          snapshot_path: str = "snapshot.pt",
                          grad_norm_clip: float = 1.0,
                          num_workers: int = 0,
                          mixup_fn: Optional[Mixup] = None,
                          seed: int = 42,
                          eval_only: bool = False,
                          loss_hyperparams: Optional[Dict] = None,
                          eq_affine_transform_params: Optional[Dict] = None,
                          use_ddp: bool = False,
                          sub_path_test: str = "",
                          dataset_name: str = "",
                          amap_saving_prob: float = 0.05,
                          class_balanced_sampling: bool = False,
                          num_samples_per_class: int = 100,
                          ) -> None:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through PDiscoTrainer class
     for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataset: A DataLoader instance for the model to be trained on.
    test_dataset: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    save_every: An integer indicating how often to save the model.
    snapshot_path: A string indicating where to save the model.
    loggers: A list of loggers to log metrics to.
    log_freq: An integer indicating how often to log metrics.
    grad_norm_clip: A float indicating the maximum gradient norm to clip to.
    enable_gradient_clipping: A boolean indicating whether to enable gradient clipping.
    mixup_fn: A Mixup instance to apply mixup to the training data.
    seed: An integer indicating the random seed to use.
    eval_only: A boolean indicating whether to only run evaluation.
    loss_hyperparams: A dictionary containing loss hyperparameters.
    eq_affine_transform_params: A dictionary containing affine transform parameters.
    use_ddp: A boolean indicating whether to use DDP.
    sub_path_test: A string indicating the sub path of the test dataset.
    dataset_name: A string indicating the name of the dataset.
    amap_saving_prob: A float indicating the probability of saving attention maps.
    class_balanced_sampling: A boolean indicating whether to use class-balanced sampling
    num_samples_per_class: An integer indicating the number of samples per class for class-balanced sampling
    @rtype: None
    """

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    if use_ddp:
        ddp_setup()

    model_trainer = PDiscoTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                                  batch_size=batch_size, optimizer=optimizer, scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                                  log_freq=log_freq,
                                  use_amp=use_amp,
                                  grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                                  mixup_fn=mixup_fn, eval_only=eval_only, loss_hyperparams=loss_hyperparams,
                                  eq_affine_transform_params=eq_affine_transform_params, use_ddp=use_ddp,
                                  sub_path_test=sub_path_test, dataset_name=dataset_name,
                                  amap_saving_prob=amap_saving_prob,
                                  class_balanced_sampling=class_balanced_sampling,
                                  num_samples_per_class=num_samples_per_class)
    if eval_only:
        model_trainer.test_only()
    else:
        model_trainer.train()
    if use_ddp:
        destroy_process_group()
