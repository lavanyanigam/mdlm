import os
import sys
import hydra
import omegaconf

# Register hydra resolvers
try:
    omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
    omegaconf.OmegaConf.register_new_resolver('device_count', lambda: 0)
    omegaconf.OmegaConf.register_new_resolver('eval', eval)
    omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except Exception:
    pass

import dataloader
import utils

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    # Override cache directory to point to the Modal persistent volume
    config.data.cache_dir = "/root/storage/mdlm_data"
    
    print("Initializing tokenizer...")
    tokenizer = dataloader.get_tokenizer(config)
    
    print("Tokenizing training and validation datasets (this can take a few minutes)...")
    # This will trigger get_dataset() for both train and validation splits,
    # which tokenizes, wraps/groups, and saves them to `/root/storage/mdlm_data`.
    dataloader.get_dataloaders(config, tokenizer)
    print("Dataset tokenization completed successfully!")

if __name__ == "__main__":
    main()
