import torch
from dalle2_pytorch import CLIP
from dataset.pusht_image_dataset import PushTPureImageDataset
import argparse
from pathlib import Path

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--decay', type=float, default=0.2, help='decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--data_path', type = str, default = '/proj/cloudrobotics-nest/users/x_ruiwa/pusht/pusht_cchi_v7_replay.zarr', help = 'The directory for training data')
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
    num_tokens=NUM_TOKENS,
    smooth_l1_loss=SMOOTH_L1_LOSS,
    num_resnet_blocks=NUM_RESNET_BLOCKS,
    kl_loss_weight=KL_LOSS_WEIGHT
)

run = wandb.init(
    project='dalle_train_clip',
    job_type='train_model',
    name='original',
    config=model_config
)

model = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 1,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 1,
    visual_image_size = 256,
    visual_patch_size = 32,
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

def configure_optimizers(learning_rate, weight_decay):
    # optimizer = torch.optim.SGD(
    #     self.parameters(),
    #     lr=lr,
    #     momentum=0.9
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=weight_decay)

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=self.num_training_steps,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=0,
        warmup_steps=2000
    )

    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

# Example training loop
num_epochs = 500  # total number of epochs
save_interval = 100  # save every 100 epochs

for epoch in range(num_epochs):
    # Your training process here
    # ...

    # Log metrics to wandb
    wandb.log({"loss": loss_value, "accuracy": accuracy_value})

    # Save model every 100 epochs
    if (epoch + 1) % save_interval == 0:
        model_path = f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

wandb.finish()

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
dataset = PushTPureImageDataset(zarr_path=args.data_path, transform=None, horizon=1)
train_dataloader = DataLoader(dataset, batch_size=args.batch_size)  # Define your own dataloader


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# add your own code to track the training progress.
for epoch in range(EPOCH):
    for batch in train_dataloader:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)