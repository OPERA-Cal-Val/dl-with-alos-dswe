# ML imports
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# math/pandas/imtools imports
import numpy as np

# misc imports
import datetime
from pathlib import Path

class sarInferenceModel(pl.LightningModule):
    def __init__(self, hparams, datasetClass=None, training_data=None, val_data=None):
        super().__init__()
        
        self.hparams.update(hparams)

        # model hyperparameters
        self.sarData = self.hparams.get("sarData", True)
        self.sarDenoisingWeight = self.hparams.get("denoisingWeight", 0.2)
        self.opticalData = self.hparams.get("opticalData", True)
        self.demData = self.hparams.get("demData", True)
        self.handData = self.hparams.get("handData", True)

        # The number of input channels will be calculated from specified inputs
        self.n_channels = 0

        # If SAR data is included in the inputs
        if self.sarData:
            self.n_channels += 2

        # If optical data is included in the inputs
        # Hansen data has 4 optical channels
        if self.opticalData:
            self.n_channels += 4

        if self.demData:
            self.n_channels += 1
        
        if self.handData:
            self.n_channels += 1

        # model input/output 
        self.in_channels = self.n_channels
        self.output_classes = self.hparams.get("output_classes")

        self.backbone = self.hparams.get("backbone", "resnet34")
        self.weights = self.hparams.get("weights", None)
        self.subtract_mean = self.hparams.get("subtract_mean", False)
        self.aux_params = self.hparams.get("aux_params", None)
        self.learning_rate = self.hparams.get("lr", 1e-3)
        self.max_epochs = self.hparams.get("max_epochs", 20)
        self.min_epochs = self.hparams.get("min_epochs", 1)
        self.patience = self.hparams.get("patience", 4)
        self.num_workers = self.hparams.get("num_workers", 8)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.output_path = self.hparams.get("output_path", "model_outputs")
        self.gpu = self.hparams.get("gpu", False)
        self.ngpus = self.hparams.get("ngpus", 0)
        self.transforms = self.hparams.get("transformations", None)
        self.channel_drop = self.hparams.get("channel_drop", False)
        self.experiment_name = self.hparams.get("experiment_name", 
                                                str(int(datetime.datetime.today().timestamp()))) + ".pt"
        
        self.class_weights = self.hparams.get("class_weights", None)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights)

        self.normalize_inputs = self.hparams.get("normalize_inputs", False)
        
        self.loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255, reduction='sum')

        # Where final model will be saved
        self.output_path = Path.cwd() / self.output_path
        self.output_path.mkdir(exist_ok=True)

        # Specify where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard_logs")
        self.log_path.mkdir(exist_ok=True)

        # Instantiate train & val datasets, model, and trainer params
        if training_data is not None:
            if 'transforms' in training_data:
                transforms = training_data['transforms']
            self.train_dataset = datasetClass(x_paths=training_data['data'], y_paths=training_data['labels'], 
            transforms=transforms, return_sar=self.sarData, denoising_weight = self.sarDenoisingWeight, 
            return_optical=self.opticalData, return_dem=self.demData, return_hand=self.handData,
            channel_drop=self.channel_drop)

        if val_data is not None:
            self.val_dataset = datasetClass(x_paths=val_data['data'], y_paths=val_data['labels'], transforms=None, 
            return_sar=self.sarData, denoising_weight = self.sarDenoisingWeight, 
            return_optical=self.opticalData, return_dem=self.demData, return_hand=self.handData,
            channel_drop=False)

        self.model = self._prepare_model()
        self.trainer_params = self._get_trainer_params()

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, idx):
        # switch on training model
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        x, y = batch
        y = y.long()

        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)

        # Calculate training loss at current step
        training_step_loss = self.loss_function(preds.clone(), y)

        # fails when loss is NaN or Inf
        assert training_step_loss == training_step_loss, f"Error with data? {np.max(x.cpu().detach().numpy())}, {np.max(y.cpu().detach().numpy())}"

        # Log values
        self.log(
            "train_loss", 
            training_step_loss,
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            rank_zero_only=False
        )

        self.log(
            "train_water_iou", 
            self.intersection_over_union(preds.clone(), y, 0),
            on_step = False, 
            on_epoch = True, # Log metric at the end of an epoch
            prog_bar = False, # Logs to progress bar
            logger = True, # Log to tensorboard
            rank_zero_only=False
        )        

        self.log(
            "train_notwater_iou", 
            self.intersection_over_union(preds.clone(), y, 1),
            on_step = False, 
            on_epoch = True, # Log metric at the end of an epoch
            prog_bar = False, # Logs to progress bar
            logger = True, # Log to tensorboard
            rank_zero_only=False
        )
        
        self.log(
            "train_water_precision",
            self.calc_precision(preds.clone(), y, 0),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )

        self.log(
            "train_water_recall",
            self.calc_recall(preds.clone(), y, 0),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )        

        return training_step_loss

    def validation_step(self, batch, idx):
        # Loss gets evaluated, but do not update model
        self.model.eval()
        torch.set_grad_enabled(False)

        x, y = batch
        y = y.long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)
        
        # Calculate Loss
        validation_step_loss = self.loss_function(preds.clone(), y)

        # Log values
        self.log(
            "val_loss", 
            validation_step_loss,
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            rank_zero_only=False
        )
        
        self.log(
            "val_water_iou", 
            self.intersection_over_union(preds.clone(), y, 0),
            on_step = False, # Log metric at each step. False because this call happens at the end of an epoch
            on_epoch = True, # Log metric at the end of an epoch
            prog_bar = False, # Logs to progress bar
            logger = True, # Log to tensorboard
            rank_zero_only=False
        )

        self.log(
            "val_notwater_iou", 
            self.intersection_over_union(preds.clone(), y, 1),
            on_step = False, # Log metric at each step. False because this call happens at the end of an epoch
            on_epoch = True, # Log metric at the end of an epoch
            prog_bar = False, # Logs to progress bar
            logger = True, # Log to tensorboard
            rank_zero_only=False
        )

        self.log(
            "water_precision",
            self.calc_precision(preds.clone(), y, 0),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )

        self.log(
            "water_recall",
            self.calc_recall(preds.clone(), y, 0),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )

        self.log(
            "notwater_precision",
            self.calc_precision(preds.clone(), y, 1),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )
        self.log(
            "notwater_recall",
            self.calc_recall(preds.clone(), y, 1),
            on_step = False,
            on_epoch = True,
            logger = True,
            rank_zero_only=False
        )   
             
        return validation_step_loss

    def validation_epoch_end(self, idx):
        # Reset metrics before next epoch
        self.intersection = 0
        self.union = 0

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train_loss",
        }  # logged value to monitor
        return [optimizer], [scheduler]

    def _get_trainer_params(self):
        # Define callback behavior
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            save_weights_only = True,
            monitor="val_loss",
            mode="min",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            patience=(self.patience),
            mode="min",
            verbose=True,
            # min_delta = 5e-4, # this value has to change based on loss reduction method.
            min_delta = 1e3
        )

        logger = pl.loggers.TensorBoardLogger(self.log_path, name=self.experiment_name)

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            # "gpus": None if not self.gpu else self.ngpus,
            "accelerator":"gpu",
            "devices": None if not self.gpu else self.ngpus,
            "strategy":"ddp", #DDPStrategy(find_unused_parameters=False), 
            "fast_dev_run": self.hparams.get("fast_dev_run", False),
            "num_sanity_val_steps": self.hparams.get("num_sanity_val_steps", 0),
            "log_every_n_steps": 10
        }
        return trainer_params

    def _prepare_model(self):
        dnn_model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=self.weights,
            in_channels=self.in_channels,
            classes=self.output_classes,
            aux_params=self.aux_params
        )

        print(f"Number of input channels and output classes : {self.in_channels, self.output_classes}")

        if self.gpu:
            assert torch.cuda.is_available, "GPU UNAVAILABLE"
            dnn_model.cuda()
        return dnn_model

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            pin_memory = False, # ??
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = False,
            drop_last=True
        )

    def fit(self):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def intersection_over_union(self, pred, target, val):
        # dim 0 corresponds to batch size
        # dim 1 corresponds to output classes. argmax be on this dim
        # will result in array of dimensions [batch_size, rows, cols]
        pred_ = torch.argmax(pred.clone().detach(), dim=1)

        def iou_helper_(pred__, target__):
            union_pixels = ((target__.eq(val) | pred__.eq(val)) & target__.ne(255)).sum()
            intersection_pixels = (target__.eq(val) & pred__.eq(val)).sum()

            iou__ = intersection_pixels/union_pixels

            if (torch.isnan(iou__) | torch.isinf(iou__)):
                return None
            else:
                return float(iou__.detach().cpu().numpy())

        sample_ious = np.array(list(map(iou_helper_, [pred_[i] for i in range(pred_.shape[0])], [target[i] for i in range(target.shape[0])])))

        return np.sum(np.where(sample_ious==None, 0, sample_ious))/np.sum(sample_ious!=None)


    def calc_precision(self, pred, target, val):
        # dim 0 corresponds to batch size
        # dim 1 corresponds to output classes. argmax be on this dim
        # will result in array of dimensions [batch_size, rows, cols]
        
        pred_ = torch.argmax(pred, dim=1)

        def precision_helper_(pred__, target__):
            valid_pixels = target__.ne(255) & pred__.eq(val)
            if valid_pixels.sum() == 0:
                return None
            else:
                return float(((pred__[valid_pixels] == target__[valid_pixels]).sum()).detach().cpu().numpy())/float(valid_pixels.sum())

        sample_precisions = np.array(list(map(precision_helper_, [pred_[i] for i in range(pred_.shape[0])], [target[i] for i in range(target.shape[0])])))

        return np.sum(np.where(sample_precisions==None, 0, sample_precisions))/np.sum(sample_precisions!=None)

    def calc_recall(self, pred, target, val):
        # dim 0 corresponds to batch size
        # dim 1 corresponds to output classes. argmax be on this dim
        # will result in array of dimensions [batch_size, rows, cols]

        pred_ = torch.argmax(pred, dim=1)

        def recall_helper_(pred__, target__):
            valid_pixels = target__.eq(val)
            if valid_pixels.sum() == 0:
                return None
            else:
                return float(((pred__[valid_pixels] == target__[valid_pixels]).sum()).detach().cpu().numpy())/float(valid_pixels.sum())

        sample_recalls = np.array(list(map(recall_helper_, [pred_[i] for i in range(pred_.shape[0])], [target[i] for i in range(target.shape[0])])))

        return np.sum(np.where(sample_recalls==None, 0, sample_recalls))/np.sum(sample_recalls!=None)
        
            
    def xe_iou_loss(self, pred, target):
        iou = self.intersection_over_union(pred, target)
        xe_loss = self.xe_loss(pred, target)

        return 0.5*xe_loss + 0.5*(1 - iou)