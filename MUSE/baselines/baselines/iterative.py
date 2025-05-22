from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
from transformers.trainer_utils import has_length, HPSearchBackend,speed_metrics, TrainOutput
from transformers.trainer import logger, _is_peft_model, TRAINER_STATE_NAME
import math
import sys
import os
import time
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init,deepspeed_load_checkpoint
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler
from transformers.integrations import hp_params
from transformers.training_args import ParallelMode
from typing import  Optional
from torch.utils.data import  RandomSampler
from accelerate import __version__ as accelerate_version
from accelerate import skip_first_batches
from accelerate.data_loader import SeedableRandomSampler
from packaging import version
from transformers.utils import is_accelerate_available, is_torch_xla_available, is_sagemaker_mp_enabled
from accelerate.utils import DistributedType
import shutil


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    gama: float = 1.0,
    unlearn_method: str = 'BLO_forget_lower'
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )
    model.resize_token_embeddings(len(tokenizer))

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")
        

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',  
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'        
    )
    
    if "BLO_forget_lower" in loss_type or "BLO_forget_upper" in loss_type:
        trainer = UnlearnerM(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type=loss_type,
            gama=gama,
            unlearn_method=unlearn_method
        )
    elif "Pen" in loss_type:
        trainer = UnlearnerM(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type=loss_type,
            gama=gama,
            unlearn_method=unlearn_method
        )
    else:
        trainer = IterativeUnlearner(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type=loss_type
        )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)
    if "BLO_forget_lower" in loss_type or "BLO_forget_upper" in loss_type or "Pen" in loss_type:
        inner_fr_v = trainer.inner_F_R
        norm_r_v = trainer.norm_R
        norm_f_v = trainer.norm_F
        torch.save(inner_fr_v, os.path.join(out_dir, "inner_F_R.pt"))
        torch.save(norm_r_v, os.path.join(out_dir, "norm_R.pt"))
        torch.save(norm_f_v, os.path.join(out_dir, "norm_F.pt"))



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.15,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        x_f, x_r = x
        
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        ### 2. Compute Loss ###
        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += (self.gama*loss_r)

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            loss += (self.gama*kl_r)

        return (loss, outputs_f) if return_outputs else loss



    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


class UnlearnerM(IterativeUnlearner):
    def __init__(self, gama, unlearn_method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_R = []
        self.norm_F = []
        self.inner_F_R = []
        self.gama = gama
        self.unlearn_method = unlearn_method 
        
        
    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
       # ### 1. Run model ###
        x_f, x_r = x
        
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        ### 2. Compute Loss return loss on f or loss on r ###
        
        loss_forget = 0
        loss_retain = 0
        

        if 'ga' in self.loss_type:
            loss_forget += -loss_f
            
        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss_forget += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss_retain += loss_r

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            loss_retain += kl_r

        return (loss_forget, loss_retain, outputs_f) if return_outputs else loss_forget, loss_retain    
    

    def training_step(self, model, inputs):
        
        
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss_forget,loss_retain = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_forget = loss_forget.mean()  # mean() to average on multi-gpu parallel training
            loss_retain = loss_retain.mean()
            
        self.accelerator.backward(loss_forget, retain_graph=True)
        # from IPython import embed; embed()
        with torch.no_grad():
            for param in model.parameters():
                if hasattr(param,'forget_grad') and param.forget_grad is not None:
                    param.forget_grad += param.grad.detach().clone()
                    param.grad = None
                else:
                    param.forget_grad = param.grad
                    param.grad = None
        self.accelerator.backward(loss_retain)
        with torch.no_grad():
            for param in model.parameters():
                if hasattr(param,'remain_grad') and param.remain_grad is not None:
                    param.remain_grad += param.grad.detach().clone()
                    param.grad = None
                    print('initialize remain_grad attribute')
                else:
                    param.remain_grad = param.grad.detach().clone()
                    param.grad = None
        loss = loss_forget + loss_retain
        return loss.detach() / self.args.gradient_accumulation_steps    
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    # Optimizer step
                    norm_r = 0
                    norm_f = 0
                    inner_r_f = 0
                    
                    tmp_upper = 0
                    tmp_lower = 0
                    
                    if self.unlearn_method == 'Pen':
                        if self.optimizer.gradient_state.sync_gradients:
                            for param in model.parameters():
                                norm_r += (param.remain_grad*param.remain_grad).sum().item()
                                norm_f += (param.forget_grad*param.forget_grad).sum().item()
                                inner_r_f += (param.remain_grad*param.forget_grad).sum().item()   
                            
                            self.norm_R.append(norm_r)
                            self.norm_F.append(norm_f)
                            self.inner_F_R.append(inner_r_f)
                    
                            with torch.no_grad():
                                for param in model.parameters():
                                    param.grad = param.forget_grad + self.gama*param.remain_grad
                                    delattr(param, 'forget_grad')
                                    delattr(param, 'remain_grad')
                    
                    elif self.unlearn_method == 'BLO_forget_lower':
                        if self.optimizer.gradient_state.sync_gradients:
                            for param in model.parameters():
                                tmp_upper += (param.remain_grad*param.forget_grad).sum().item()
                                tmp_lower += (param.forget_grad*param.forget_grad).sum().item()
                                norm_r += (param.remain_grad*param.remain_grad).sum().item()
                                
                            self.norm_R.append(norm_r)
                            self.norm_F.append(tmp_lower)
                            self.inner_F_R.append(tmp_upper)
                                
                            xi = self.gama - tmp_upper/tmp_lower
                                
                            for param in model.parameters():
                                param.grad = param.remain_grad + xi*param.forget_grad
                                delattr(param, 'forget_grad')
                                delattr(param, 'remain_grad')
                                
                            
                    elif self.unlearn_method == 'BLO_forget_upper':
                        if self.optimizer.gradient_state.sync_gradients:
                            for param in model.parameters():
                                tmp_upper += (param.remain_grad*param.forget_grad).sum().item()
                                tmp_lower += (param.remain_grad*param.remain_grad).sum().item()
                                norm_f += (param.forget_grad*param.forget_grad).sum().item()
                                
                            self.norm_R.append(tmp_lower)
                            self.norm_F.append(norm_f)
                            self.inner_F_R.append(tmp_upper)
                                
                            xi = self.gama - tmp_upper/tmp_lower
                                    
                            for param in model.parameters():
                                param.grad = param.forget_grad + xi*param.remain_grad
                                delattr(param, 'forget_grad')
                                delattr(param, 'remain_grad')
                    else:
                        raise NotImplementedError("Cannot infer the given unlearn method.")
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            # elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            #     dist.barrier()
            # elif is_sagemaker_mp_enabled():
            #     smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)     
