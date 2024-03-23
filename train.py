from fire import Fire
import yaml
from dm import Unet, GaussianDiffusion, Trainer
from planetAI.src.data.utils import PlanetConfig

def main(
        config_file: str = None,
        image_size: int = 256,
        dim: int = 256,
        num_classes: int = 10,
        dim_mults: str = '1 2 4',
        channels: int = 4,
        resnet_block_groups: int = 2,
        block_per_layer: int = 2,
        timesteps: int = 1000,
        sampling_timesteps: int = 100,
        batch_size: int = 8,
        lr: float = 1e-4,
        train_num_steps: int = 2500000,
        save_sample_every: int = 25000,
        gradient_accumulate_every: int = 1,
        save_loss_every: int = 100,
        num_samples: int = 4,
        num_workers: int = 8,
        results_folder: str = './results/run_name',
        milestone: int = None,
        data_folder: str = '../terrain-ml/planetAI/data'
):
    planet_cfg = PlanetConfig(data_dir=data_folder)
    num_classes = planet_cfg.combined_classes()

    dim_mults=[int(mult) for mult in dim_mults.split(' ')]

    if config_file:
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
        for key in config.keys():
            locals().update(config[key])
    
    z_size=image_size//8
    
    unet = Unet(
            dim=dim,
            num_classes=num_classes,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups = resnet_block_groups,
            block_per_layer=block_per_layer,
        )

    model = GaussianDiffusion(
            unet,
            image_size=z_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type='l2')

    trainer = Trainer(
            model,
            train_batch_size=batch_size,
            train_lr=lr,
            train_num_steps=train_num_steps,
            save_and_sample_every=save_sample_every,
            gradient_accumulate_every=gradient_accumulate_every,
            save_loss_every=save_loss_every,
            num_samples=num_samples,
            num_workers=num_workers,
            results_folder=results_folder,
            data_folder=data_folder)

    if milestone:
        trainer.load(milestone)
        
    trainer.train()

if __name__=='__main__':
    Fire(main)