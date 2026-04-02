import logging

import torch
from torch import nn

from fedml.core import ClientTrainer


class VeriFLTrainer(ClientTrainer):
    def __init__(self, model, args):
        self.cpu_transfer = bool(getattr(args, "cpu_transfer", True))
        super().__init__(model, args)

    def get_model_params(self):
        if self.cpu_transfer:
            return self.model.cpu().state_dict()
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def train(self, train_data, device, args):
        # Phase 2: 攻击注入由 FedML 内置机制负责；
        # label_flipping 在 update_dataset() 中处理，模型攻击在聚合器 on_before_aggregation() 中处理。
        model = self.model
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(getattr(args, "learning_rate", 0.01)),
            momentum=float(getattr(args, "momentum", 0.9)),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )
        logging.info(
            "Client %s optimizer | lr=%s momentum=%s weight_decay=%s",
            self.id,
            optimizer.defaults["lr"],
            optimizer.defaults["momentum"],
            optimizer.defaults["weight_decay"],
        )
        epoch_loss = []
        for epoch in range(int(getattr(args, "epochs", 1))):
            batch_losses = []
            for batch_idx, (images, labels) in enumerate(train_data):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                if batch_idx % 20 == 0:
                    logging.info(
                        "Client %s | Epoch %s | Batch %s/%s | Loss %.6f",
                        self.id,
                        epoch + 1,
                        batch_idx + 1,
                        len(train_data),
                        loss.item(),
                    )
            if batch_losses:
                epoch_loss.append(sum(batch_losses) / len(batch_losses))
        if epoch_loss:
            logging.info(
                "Client %s finished local training with mean loss %.6f",
                self.id,
                sum(epoch_loss) / len(epoch_loss),
            )

    def test(self, test_data, device, args):
        if test_data is None:
            return None
        model = self.model
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        metrics = {
            "test_correct": 0,
            "test_loss": 0.0,
            "test_total": 0,
        }
        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                _, predicted = torch.max(logits, 1)
                metrics["test_correct"] += predicted.eq(labels).sum().item()
                metrics["test_loss"] += loss.item() * labels.size(0)
                metrics["test_total"] += labels.size(0)
        return metrics
