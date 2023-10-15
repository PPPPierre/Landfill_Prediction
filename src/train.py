import os
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from .utils.checkpoint import save_checkpoint, load_checkpoint
from .evaluate import evaluate
from .optim import create_optimizer_with_scheduler_from_cfg
from .utils.checkpoint import save_checkpoint, load_checkpoint
from .loss import BCEWithLogitsLoss

checkpoint_path = "checkpoint.pth"

def train(
        config: dict, 
        time_stamp: str,
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        model: torch.nn.Module, 
        device: torch.device, 
        ):

    train_cfg = config['train']
    test_cfg = config['test']

    logger = logging.getLogger("__main__")
    model.train()
    
    # Define criterion
    criterion = BCEWithLogitsLoss()

    # Defining optimizer and scheduler
    optimizer, scheduler = create_optimizer_with_scheduler_from_cfg(model.parameters(), train_cfg)

    # Set save dir
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = train_cfg.get("result_dir", None)
    if save_dir:
        save_dir = os.path.join(root_path, save_dir)
    else:
        save_dir = os.path.join(root_path, f"results/{time_stamp}")

    # Resume training
    resume_ckpt = train_cfg.get("resume_ckpt", "")
    start_epoch = 0
    if torch.cuda.is_available() and os.path.exists(resume_ckpt):
        start_epoch = load_checkpoint(model, optimizer, resume_ckpt)
    scheduler.set_last_epoch(start_epoch)
    
    # Define training params
    num_epochs = train_cfg['epochs']
    start_time = time.time()

    # MLflow trace
    for epoch in range(start_epoch, num_epochs):
        # Train
        total_loss = 0.0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()

            # set zero gradient
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimization
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            if i % 5 == 4:  # output verbose each ten steps
                elapsed_time = time.time() - start_time
                estimated_time = (elapsed_time / (epoch * len(train_loader) + i + 1)) * (num_epochs * len(train_loader))
                time_remaining = estimated_time - elapsed_time
                logger.info(f"[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 10:.6f}, elapsed time: {elapsed_time:.2f} seconds, estimated total time: {estimated_time:.2f} seconds, time remaining: {time_remaining:.2f} seconds")
                running_loss = 0.0

        average_loss = total_loss / len(train_loader.dataset)
        
        # Evaluate
        metrics = evaluate(test_cfg, model, test_loader, criterion, device)
        logger.info(f"[Epoch {epoch + 1}] learning rate: {scheduler.current_lr:.6f}, train loss: {average_loss:.6f}, AUC: {metrics['AUC']:.3f}, accuracy: {metrics['accuracy']:.3f}, precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, f1_score: {metrics['f1_score']:.3f}")
        
        # Save model
        save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
        save_checkpoint(epoch, model, optimizer, save_path)

        scheduler.step()

