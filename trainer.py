import argparse
import os
import logging
import time

import torch.cuda
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from config import Config

from model import model
# from transformers import AdamW

from itertools import zip_longest

from utils import RNNDataLoader, CNNDataLoader, make_dir


class Trainer:
    def __init__(self, config, train_iter=None, valid_iter=None, test_iter=None):
        self.config = config
        self.model_type = config.model_type
        # BiLSTM, BiLSTMLAN, CNN
        self.model = getattr(model, self.model_type)(config)
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.log_dir = os.getcwd() + '/log/' + self.model_type
        make_dir(self.log_dir)
        self.training_time = 0

        self.logger.setLevel(logging.INFO)
        self.filehandler = logging.FileHandler(os.path.join(self.log_dir, os.path.basename(self.config.config_path).split('.')[0] + '.log'))
        self.filehandler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]: %(message)s'))
        self.logger.addHandler(self.filehandler)

        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        # BERT-like linear warmup scheduler
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.lr,
                                    pct_start=0.1, epochs=self.config.epoch,
                                    steps_per_epoch=len(train_iter.contexts) // self.config.batch_size + 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_idx = self.model.tag2idx["[PAD]"]

        self.model.to(self.device)

        # ignore index ?
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=config.label_smoothing)
        self.criterion.to(self.device)


    def train(self):
        print(self.model)
        self.logger.info(self.model)
        print(f"Number of model's trainable parameters: {self.model.count_params():,}")
        self.logger.info(f"Number of model's trainable parameters: {self.model.count_params():,}")
        start = time.time()
        best_val_loss = float('inf')
        early_stop_cnt = 0
        end_epoch = 0
        for epoch in tqdm(range(1, self.config.epoch + 1), desc='epochs'):
            self.model.train()
            epoch_loss = 0
            step = 0

            for batch in self.train_iter:
                text, label, mask, *_ = batch
                text = text.to(self.device)
                label = label.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(text, mask)
                logits = logits.view(-1, logits.shape[-1])
                label = label.view(-1)
                loss = self.criterion(logits, label)

                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                step += 1

                #if step % 2000 == 0:
                #    print(f'Step Loss: {(epoch_loss/step):.3f}')

            train_loss = epoch_loss / step
            valid_loss = self.eval()
            syl_acc, morph_acc = self.infer_cnn() if self.model_type == "CNN" else self.infer_rnn()

            self.logger.info(f'Epoch: {epoch:02d} | Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}')
            self.logger.info(f'          | syl_acc: {syl_acc:.5f} % , morph_acc: {morph_acc:.5f} %')

            print(f'Epoch: {epoch:02d} | Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}')
            print(f'          | syl_acc: {syl_acc:.5f} % , morph_acc: {morph_acc:.5f} %')

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt == self.config.early_stop:
                end_epoch = epoch
                break

            if self.model_type == "CNN":
                checkpoint_name = f'hidden_{self.config.hidden_dim}_ws{getattr(self.config, self.model_type)["window_size"]}_ks{getattr(self.config, self.model_type)["max_kernel_size"]}'
            else:
                checkpoint_name = f'l{getattr(self.config, self.model_type)["num_layers"]}_hidden_{self.config.hidden_dim}'
            self.save_model(os.path.join(os.getcwd(), self.config.checkpoint_path, self.model_type, checkpoint_name), {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": valid_loss
            })
        end = time.time()
        self.logger.info(f"total training time: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")
        self.logger.info(f"training time per epoch: {time.strftime('%H:%M:%S', time.gmtime(end-start))} / {end_epoch} = {time.strftime('%H:%M:%S', time.gmtime((end-start)/end_epoch))}")


    def eval(self):
        self.model.eval()
        loss = 0
        step = 0
        for batch in self.valid_iter:
            text, label, mask, *_ = batch
            text = text.to(self.device)
            label = label.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                logits = self.model(text, mask)
                logits = logits.view(-1, logits.shape[-1])
                label = label.view(-1)
                loss += self.criterion(logits, label).item()
                step += 1

        return loss / step

    def infer_rnn(self):
        self.model.eval()
        total_syllable_count = 0
        correct_syllable_count = 0
        total_morph_count = 0
        correct_morph_count = 0

        for batch in self.test_iter:
            text, label, mask, raw_sen = batch
            text = text.to(self.device)
            label = label.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                predict = self.model(text, mask)
                predict = predict.argmax(-1)

                for infer, gold, raw_ in zip(predict, label, raw_sen):
                    try:
                        infer = infer.tolist()
                        gold = gold.tolist()
                        infer = infer[1:infer.index(self.model.tag2idx["[EOS]"])]
                        gold = gold[1:gold.index(self.model.tag2idx["[EOS]"])]
    
                        for i, g in zip_longest(infer, gold, fillvalue=-1):
                            if g == -1:
                                break

                            total_syllable_count += 1
                            g_tag = self.model.idx2tag[str(g)]
                            i_tag = self.model.idx2tag[str(i)] if i != -1 else ""

                            morph_count = 1
    
                            if ':' in g_tag:
                                morph_len = len(g_tag.split(":"))
                                if morph_len > 2:
                                    morph_count = morph_len - 1
    
                            total_morph_count += morph_count
    
                            if i == g:
                                correct_syllable_count += 1
                                correct_morph_count += morph_count
                        
                        ### should be updated morph accuracy later
                            else:
                                for i_morph, g_morph in zip_longest(i_tag.split(':')[:-1], g_tag.split(':')[:-1]):
                                    if g_morph == None:
                                        break
                                    if i_morph == g_morph:
                                        correct_morph_count += 1
                        
                    except ValueError:
                        pass
                        # print(raw_)
                        # print(f'infer: {infer}')
                        # print(f'gold: {gold}')

        return correct_syllable_count * 100 / total_syllable_count, correct_morph_count * 100 / total_morph_count

    def infer_cnn(self):
        self.model.eval()
        total_syllable_count = 0
        correct_syllable_count = 0
        total_morph_count = 0
        correct_morph_count = 0

        for batch in self.test_iter:
            text, label, mask = batch
            text = text.to(self.device)
            label = label.to(self.device).view(-1)
            mask = mask.to(self.device)

            with torch.no_grad():
                predict = self.model(text, mask)
                predict = predict.argmax(-1)

                for infer, gold in zip(predict, label):
                    total_syllable_count += 1

                    gold_tag = self.model.idx2tag[str(gold.item())]
                    infer_tag = self.model.idx2tag[str(infer.item())]

                    morph_count = 1

                    if ':' in gold_tag:
                        morph_len = len(gold_tag.split(":"))
                        if morph_len > 2:
                            morph_count = morph_len - 1

                    total_morph_count += morph_count

                    if infer.item() == gold.item():
                        correct_syllable_count += 1
                        correct_morph_count += morph_count
                    else:
                        for i_morph, g_morph in zip_longest(infer_tag.split(":")[:-1], gold_tag.split(":")[:-1]):
                            if g_morph == None:
                                break
                            if i_morph == g_morph:
                                correct_morph_count += 1

        return correct_syllable_count * 100 / total_syllable_count, correct_morph_count * 100 / total_morph_count


    def save_model(self, save_path, state_dict):
        make_dir(save_path)
        torch.save(state_dict, os.path.join(save_path, 'checkpoint_epoch_' + str(state_dict['epoch']) + f'_{state_dict["loss"]:.3f}' + '.pt'))


def main(args):
    config = Config(args.config)

    if config.model_type == "CNN":
        train_iter = CNNDataLoader(config, 'train')
        valid_iter = CNNDataLoader(config, 'valid')
        test_iter = CNNDataLoader(config, 'test')
        trainer = Trainer(config, train_iter=train_iter, valid_iter=valid_iter, test_iter=test_iter)
        trainer.train()
    else:
        train_iter = RNNDataLoader(config, 'train')
        valid_iter = RNNDataLoader(config, 'valid')
        test_iter = RNNDataLoader(config, 'test')
        trainer = Trainer(config, train_iter=train_iter, valid_iter=valid_iter, test_iter=test_iter)
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path", required=True)

    args = parser.parse_args()
    main(args)
