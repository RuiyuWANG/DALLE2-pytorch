import torch
import os
from dalle2_pytorch import CLIP
from dataset.pusht_image_dataset import PushTCLIPDataset
import argparse
from pathlib import Path

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--decay', type=float, default=0.2, help='decay rate')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--save_interval', type=int, default=100, help='save interval')
parser.add_argument('--val_interval', type=int, default=100, help='val interval')
parser.add_argument('--val_ratio', type=float, default=0.03, help='val ratio')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
parser.add_argument('--t_max', type=int, default=1000, help='for learning rate scheduler')
parser.add_argument('--dim_data', type=int, default=64, help='dimension of text and image emb')
parser.add_argument('--dim_latent', type=int, default=256, help='dimension of latent')
parser.add_argument('--data_path', type = str, default = '/proj/cloudrobotics-nest/users/x_ruiwa/pusht/pusht_cchi_v7_replay.zarr', help = 'The directory for training data')
parser.add_argument('--save_dir', type=str, default='/proj/cloudrobotics-nest/users/x_ruiwa/running_clip/train_clip', help='The directory for saving checkpoints')
args = parser.parse_args()

def save_model(path):
    # save_obj = {
    #     'hparams': vae_params,
    # }
    save_obj = dict()
    cp_path = Path(path)
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = str(path_sans_extension) + '-ds-cp'

    distr_vae.save_checkpoint(cp_dir, client_state=save_obj)

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

model_config = dict(
    dim_data = args.dim_data,
    dim_latent = args.dim_latent
)

run = wandb.init(
    project='dalle_train_clip',
    job_type='train_model',
    name='transformer_transformer',
    config=model_config
)

model = CLIP(
    dim_text = args.dim_data,
    dim_image = args.dim_data,
    dim_latent = args.dim_latent,
    num_text_tokens = 640,
    text_enc_depth = 1,
    text_seq_len = 2,
    text_heads = 8,
    visual_enc_depth = 1,
    visual_image_size = 96,
    visual_patch_size = 8,
    visual_heads = 8,
    use_all_token_embeds = True,            # whether to use fine-grained contrastive learning (FILIP)
    decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
    extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_visual_ssl = True,                  # whether to do self supervised learning on images
    visual_ssl_type = 'simclr',             # can be either 'simclr' or 'simsiam', depending on using DeCLIP or SLIP
    use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
    text_ssl_loss_weight = 0.05,            # weight for text MLM loss
    image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
).cuda()


train_dataset = PushTCLIPDataset(zarr_path=args.data_path, horizon=1, val_ratio=args.val_ratio)
val_dataset = train_dataset.get_validation_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=args.weight_decay)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)

for epoch in range(args.num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()

        images, texts = batch
        images = torch.moveaxis(images, -1, 1) / 255.
        texts = texts.reshape(-1, 2).to(torch.long)

        images = images.to(args.device)
        texts = texts.to(args.device)

        loss = model(
            texts,
            images,
            freeze_image_encoder=False,
            return_loss=True
        )

        loss.backward()
        optimizer.step()

        logs = {
            'epoch': epoch,
            'loss': loss.item(),
            'lr': lr_scheduler.get_lr()[0]
        }

        model.eval()
        if (epoch + 1) % args.val_ratio == 0:
            with torch.no_grad():
                val_losses = list()
                for batch in val_dataloader:
                    images, texts = batch
                    images = images.to(args.device)
                    texts = texts.to(args.device)
                    loss = model(
                        texts,
                        images,
                        freeze_image_encoder=False,
                        return_loss=True
                    )
                    val_losses.append(loss)
                logs['val_loss'] = val_losses
                print('val loss: ', sum(val_losses), 'at epoch', epoch)

                del val_losses
                del batch

        # Save model every 100 epochs
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

        wandb.log(logs)
        model.train()
    lr_scheduler.step()
wandb.finish()