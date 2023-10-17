import os
import sys
import time
import logging

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.dataset import get_dataloader_from_cfg
from src.model import get_model_from_cfg
from src.optim import create_optimizer_with_scheduler_from_cfg
from src.loss import BCEWithLogitsLoss
from src.evaluate import evaluate
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.checkpoint import save_checkpoint, load_checkpoint

checkpoint_path = "checkpoint.pth"

def train(config: dict, save_dir: str):

    # Set logger
    logger = logging.getLogger("__main__")
    logger.info(f"Config: {config}")

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Loading cfgs
    train_cfg = config["train"]
    test_cfg = config["test"]

    # Loading data
    train_set, train_loader = get_dataloader_from_cfg(train_cfg["data"])
    test_set, test_loader = get_dataloader_from_cfg(test_cfg["data"])
    logger.info(f"Training data: {len(train_set)}, Testing data: {len(test_set)}")

    # Init model
    model_cfg = config["model"]
    model = get_model_from_cfg(model_cfg)
    model = model.to(device)
    model.train()
    
    # Define criterion
    criterion = BCEWithLogitsLoss()

    # Defining optimizer and scheduler
    optimizer, scheduler = create_optimizer_with_scheduler_from_cfg(model.parameters(), train_cfg)

    # Resume training
    resume_ckpt = train_cfg.get("resume_ckpt", "")
    start_epoch = 0
    if torch.cuda.is_available() and os.path.exists(resume_ckpt):
        start_epoch = load_checkpoint(model, optimizer, resume_ckpt)
    scheduler.set_last_epoch(start_epoch)
    logger.info(f"{model}")
    
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
        logger.info(f"[Epoch {epoch + 1}] train loss: {average_loss:.6f}, AUC: {metrics['AUC']:.3f} with optimal threashold: {metrics['optimal_threshold']:.3f}, accuracy: {metrics['accuracy']:.3f}, precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, f1_score: {metrics['f1_score']:.3f}")
        
        # Save model
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pth')
        save_checkpoint(epoch, model, optimizer, save_path)
        scheduler.step()

    # Save the last epoch
    save_path = os.path.join(checkpoint_dir, 'last.pth')
    save_checkpoint(epoch, model, optimizer, save_path)
    logger.info("$END_OF_LOGS$")

