# Original LoRA train script by @Akegarasu ; rewritten in Python by LJRE.
import subprocess
import os
import folder_paths
import random
from comfy import model_management
import torch

from .utils.formaters import value_formater

# Default settings
is_v2_model = 0
parameterization = 0

reg_data_dir = ""

# Network settings 
network_module = "networks.lora"
network_weights = ""
network_dim = 32 
network_alpha = 32

# Default resolution
resolution = "512,512"

train_unet_only = 0
train_text_encoder_only = 0
stop_text_encoder_training = 0

noise_offset = 0
keep_tokens = 0
min_snr_gamma = 0

lr = "1e-4"
unet_lr = "1e-4"
text_encoder_lr = "1e-5"
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 0
lr_restart_cycles = 1


optimizer_type = "AdamW8bit"

# TODO: not sure if these are necessary
save_model_as = "safetensors"

save_state = 0
resume = ""

# Other settings
min_bucket_reso = 256
max_bucket_reso = 1584
persistent_data_loader_workers = 1

# Acceleration settings # TODO: not sure if these are necessary
multi_gpu = 0
lowram = 0

# LyCORIS
algo = "lora"
conv_dim = 4 
conv_alpha = 4
dropout = "0"

# Settings for wandb
use_wandb = 0 
wandb_api_key = "" 
log_tracker_name = ""



logging_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
log_prefix = ''
caption_extension = '.txt'


os.environ['HF_HOME'] = "huggingface"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"
ext_args = []
launch_args = []
flux_args = []

def GetTrainScript(script_name:str):
    # Current file directory from __file__
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    sd_script_dir = os.path.join(current_file_dir, "sd-scripts")
    train_script_path = os.path.join(sd_script_dir, f"{script_name}.py")
    return train_script_path, sd_script_dir

class LoraTraininginComfy:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "model_type": (["sd1.5", "sd2.0", "sdxl"], ),
            "resolution_width": ("INT", {"default":512, "step":64}),
            "resolution_height": ("INT", {"default":512, "step":64}),
            #"theseed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "data_path": ("STRING", {"default": "Insert path of image folders"}),
			"batch_size": ("INT", {"default": 1, "min":1}),
            "max_train_epoches": ("INT", {"default":10, "min":1}),
            "save_every_n_epochs": ("INT", {"default":10, "min":1}),
            #"lr": ("INT": {"default":"1e-4"}),
            #"optimizer_type": ("STRING", {["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"]}),
            "output_name": ("STRING", {"default":'Desired name for LoRA.'}),
            "clip_skip": ("INT", {"default":2, "min":1}),
            "output_dir": ("STRING", {"default":'models/loras'}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "loratraining"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    
    def loratraining(self, ckpt_name, resolution_width, resolution_height, model_type, data_path, batch_size, max_train_epoches, save_every_n_epochs, output_name, clip_skip, output_dir):
        #free memory first of all
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
            
        print(model_management.current_loaded_models)
        
        #TODO: not sure if these are necessary
        #loadedmodel = model_management.LoadedModel()
        #loadedmodel.model_unload(self, current_loaded_models)
        #=======================================================
        
        #transform backslashes into slashes for user convenience.
        train_data_dir = data_path.replace( "\\", "/")
        if data_path == "Insert path of image folders":
            raise ValueError("Please insert the path of the image folders.")
        if output_name == 'Desired name for LoRA.': 
            raise ValueError("Please insert the desired name for LoRA.")
        train_script_name = "train_network"

        #generates a random seed
        theseed = random.randint(0, 2^32-1)
        
        if multi_gpu:
            launch_args.append("--multi_gpu")

        if lowram:
            ext_args.append("--lowram")

        if model_type == "sd2.0":
            ext_args.append("--v2")
        elif model_type == "sd1.5":
            ext_args.append(f"--clip_skip={clip_skip}")
        elif model_type == "sdxl":
            train_script_name = "sdxl_train_network"
        
        resolution = f"{resolution_width},{resolution_height}"
        #input()nd("--network_train_text_encoder_only")

        if network_weights:
            ext_args.append(f"--network_weights={network_weights}")

        if reg_data_dir:
            ext_args.append(f"--reg_data_dir={reg_data_dir}")

        if optimizer_type:
            ext_args.append(f"--optimizer_type={optimizer_type}")

        if optimizer_type == "DAdaptation":
            ext_args.append("--optimizer_args")
            ext_args.append("decouple=True")

        if network_module == "lycoris.kohya":
            ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        if noise_offset != 0:
            ext_args.append(f"--noise_offset={noise_offset}")

        if stop_text_encoder_training != 0:
            ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")

        if save_state == 1:
            ext_args.append("--save_state")

        if resume:
            ext_args.append(f"--resume={resume}")

        if min_snr_gamma != 0:
            ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

        if persistent_data_loader_workers:
            ext_args.append("--persistent_data_loader_workers")

        if use_wandb == 1:
            ext_args.append("--log_with=all")
            if wandb_api_key:
                ext_args.append(f"--wandb_api_key={wandb_api_key}")
            if log_tracker_name:
                ext_args.append(f"--log_tracker_name={log_tracker_name}")
        else:
            ext_args.append("--log_with=tensorboard")

        launchargs=' '.join(launch_args)
        extargs=' '.join(ext_args)

        pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        #Looking for the training script.
        progpath = os.getcwd()
        nodespath=''
        sd_script_dir=''
        for dirpath, dirnames, filenames in os.walk(progpath):
             if 'sd-scripts' in dirnames:
               nodespath = dirpath + f'/sd-scripts/{train_script_name}.py'
               sd_script_dir = dirpath + '/sd-scripts'
               print(nodespath)

        nodespath = nodespath.replace( "\\", "/")
        command = "python -m accelerate.commands.launch " + launchargs + f'--num_cpu_threads_per_process=8 "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="{logging_dir}" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --save_every_n_epochs={save_every_n_epochs} --mixed_precision="fp16" --save_precision="fp16" --seed={theseed} --cache_latents --prior_loss_weight=1 --max_token_length=225 --caption_extension=".txt" --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} --xformers --shuffle_caption ' + extargs
        subprocess.run(command, shell=True, cwd=sd_script_dir)
        print("Train finished")
        return ()

class LoraTraininginComfyAdvanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "unet": (folder_paths.get_filename_list("unet"), ),
            "clip_l": (folder_paths.get_filename_list("clip"), ),
            "t5xxl": (folder_paths.get_filename_list("clip"), ),
            "vae": (folder_paths.get_filename_list("vae"), ), # TODO: vae as "--ae"
            "flux_vram": (["12GB", "16GB", "24GB"], ),
            "model_type": (["sd1.5", "sd2.0", "sdxl", "flux1.0"], ),
            "networkmodule": (["networks.lora", "lycoris.kohya", "networks.lora_flux"], ),
            "networkdimension": ("INT", {"default": 32, "min":0}),
            "networkalpha": ("INT", {"default":32, "min":0}),
            "num_processes": ("INT", {"default":1, "min":1}),
            "gpu_ids": ("STRING", {"default":"0"}),
            "resolution_width": ("INT", {"default":512, "step":64}),
            "resolution_height": ("INT", {"default":512, "step":64}),
            "data_path": ("STRING", {"default": "Insert path of image folders"}),
            "reg_data_dir": ("STRING", {"default": "Insert path of reg image folders"}),
			"batch_size": ("INT", {"default": 1, "min":1}),
            "max_train_epoches": ("INT", {"default":10, "min":1}),
            "save_every_n_epochs": ("INT", {"default":10, "min":1}),
            "keeptokens": ("INT", {"default":0, "min":0}),
            "minSNRgamma": ("FLOAT", {"default":0, "min":0, "step":0.1}),
            "learningrateText": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learningrateUnet": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learningRateScheduler": (["cosine_with_restarts", "linear", "cosine", "polynomial", "constant", "constant_with_warmup"], ),
            "lrRestartCycles": ("INT", {"default":1, "min":1}),
            "optimizerType": (["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"], ),
            "output_name": ("STRING", {"default":'Desired name for LoRA.'}),
            "algorithm": (["lora","loha","lokr","ia3","dylora", "locon"], ),
            "mixed_precision": (["fp16", "fp32", "bf16"], ),
            "save_precision": (["fp16", "fp32", "bf16"], ),
            "networkDropout": ("FLOAT", {"default": 0, "step":0.1}),
            "clip_skip": ("INT", {"default":2, "min":1}),
            "output_dir": ("STRING", {"default":'models/loras'}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "loratraining"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def loratraining(self, ckpt_name, unet, clip_l, t5xxl, vae, flux_vram, model_type, networkmodule, networkdimension, networkalpha, num_processes, gpu_ids, resolution_width, resolution_height, data_path, reg_data_dir, batch_size, max_train_epoches, save_every_n_epochs, keeptokens, minSNRgamma, learningrateText, learningrateUnet, learningRateScheduler, lrRestartCycles, optimizerType, output_name, algorithm, networkDropout, mixed_precision, save_precision, clip_skip, output_dir):
        #free memory first of all
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
        
        #TODO: not sure if these are necessary    
        #print(model_management.current_loaded_models)
        #loadedmodel = model_management.LoadedModel()
        #loadedmodel.model_unload(self, current_loaded_models)
        #=======================================================
        
        #transform backslashes into slashes for user convenience.
        train_data_dir = data_path.replace( "\\", "/")
        if data_path == "Insert path of image folders":
            raise ValueError("Please insert the path of the image folders.")

        if output_name == 'Desired name for LoRA.': 
            raise ValueError("Please insert the desired name for LoRA.")
        

        # #ADVANCED parameters initialization
        network_module=networkmodule
        network_dim=32
        network_alpha=32
        resolution = "512,512"
        keep_tokens = 0
        min_snr_gamma = 0
        unet_lr = "1e-4"
        text_encoder_lr = "1e-5"
        lr_scheduler = "cosine_with_restarts"
        lr_restart_cycles = 0
        optimizer_type = "AdamW8bit"
        algo= "lora"
        dropout = 0.0
        train_script_name = "train_network"
        raw_args = '--prior_loss_weight=1 --max_token_length=225 --xformers --split_mode --caption_extension=".txt"'
        cache_args = '--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --cache_latents_to_disk'
        
        
        if model_type == "sd1.5":
            ext_args.append(f"--clip_skip={clip_skip}")
        elif model_type == "sd2.0":
            ext_args.append("--v2")
        elif model_type == "sdxl":
            train_script_name = "sdxl_train_network"
        elif model_type == "flux1.0":
            # TODO: add flux_vram option if
            flux_def = ' --persistent_data_loader_workers --max_data_loader_n_workers 2 --sdpa --gradient_checkpointing --timestep_sampling sigmoid --model_prediction_type raw --optimizer_type adafactor --network_args "train_blocks=single" --network_train_unet_only --fp8_base --highvram --guidance_scale 1.0 --loss_type l2'
            clip_l_path = folder_paths.get_full_path("clip", clip_l)
            t5xxl_path = folder_paths.get_full_path("clip", t5xxl)
            vae_path = folder_paths.get_full_path("vae", vae)
            flux_args.append(f"--clip_l {clip_l_path} --t5xxl {t5xxl_path} --ae {vae_path}")
            if flux_vram == "12GB":
                ext_args.append('--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --split_mode --network_args "train_blocks=single"')
            elif flux_vram == "16GB":
                ext_args.append('--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"')
            elif flux_vram == "24GB":
                pass
            ext_args.append(flux_def)
            train_script_name = "flux_train_network"
        
        network_module = networkmodule
        network_dim = networkdimension
        network_alpha = networkalpha
        resolution = f"{resolution_width},{resolution_height}"
        
        text_encoder_lr = value_formater(learningrateText)        
        unet_lr = value_formater(learningrateUnet)
        
        keep_tokens = keeptokens
        min_snr_gamma = minSNRgamma
        lr_scheduler = learningRateScheduler
        lr_restart_cycles = lrRestartCycles
        optimizer_type = optimizerType
        algo = algorithm
        dropout = f"{networkDropout}"

        
        #Generates a random seed
        theseed = random.randint(0, 2^32-1)
        
        if multi_gpu:
            launch_args.append("--multi_gpu")
        
        if network_module == "lycoris.kohya":
            ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        if lowram:
            ext_args.append("--lowram")

        if parameterization:
            ext_args.append("--v_parameterization")

        if train_unet_only:
            ext_args.append("--network_train_unet_only")

        if train_text_encoder_only:
            ext_args.append("--network_train_text_encoder_only")

        if network_weights:
            ext_args.append(f"--network_weights={network_weights}")

        if reg_data_dir:
            ext_args.append(f"--reg_data_dir={reg_data_dir}")

        if optimizer_type:
            ext_args.append(f"--optimizer_type={optimizer_type}")

        if optimizer_type == "DAdaptation":
            ext_args.append("--optimizer_args")
            ext_args.append("decouple=True")

        if network_module == "lycoris.kohya":
            ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        if noise_offset != 0:
            ext_args.append(f"--noise_offset={noise_offset}")

        if stop_text_encoder_training != 0:
            ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")

        if save_state == 1:
            ext_args.append("--save_state")

        if resume:
            ext_args.append(f"--resume={resume}")

        if min_snr_gamma != 0:
            ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

        if persistent_data_loader_workers:
            ext_args.append("--persistent_data_loader_workers")

        if use_wandb == 1:
            ext_args.append("--log_with=all")
            if wandb_api_key:
                ext_args.append(f"--wandb_api_key={wandb_api_key}")
            if log_tracker_name:
                ext_args.append(f"--log_tracker_name={log_tracker_name}")
        else:
            ext_args.append("--log_with=tensorboard")

        launchargs=' '.join(launch_args)
        extargs=' '.join(ext_args)
        fluxargs=' '.join(flux_args)
        
        if model_type == "flux1.0":
            pretrained_model = folder_paths.get_full_path("unet", unet)
        else:
            pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        
        #Looking for the training script.
        nodespath, sd_script_dir = GetTrainScript(script_name=train_script_name)
        print(nodespath)
        print(sd_script_dir)
        test_args = ''
        # command = f"python -m accelerate.commands.launch " + launchargs + f'--num_cpu_threads_per_process=8 --gpu_ids="{gpu_ids}" --num_processes={num_processes} "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} ' + fluxargs + f' --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="{logging_dir}" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --save_every_n_epochs={save_every_n_epochs} --mixed_precision={mixed_precision} --save_precision={save_precision} --seed={theseed} --cache_latents --prior_loss_weight=1 --max_token_length=225 --caption_extension=".txt" --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} --xformers --shuffle_caption ' + extargs
        # command_ = f"python -m accelerate.commands.launch " + launchargs + f'--num_cpu_threads_per_process=8 --gpu_ids="{gpu_ids}" --num_processes={num_processes} "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} ' + fluxargs + f' --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="{logging_dir}" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --save_every_n_epochs={save_every_n_epochs} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --mixed_precision={mixed_precision} --save_precision={save_precision} --seed={theseed} --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} --prior_loss_weight=1 --max_token_length=225 --caption_extension=".txt" ' + extargs
        command =  f'python -m accelerate.commands.launch ' + launchargs + f'--num_cpu_threads_per_process=8 --gpu_ids="{gpu_ids}" --num_processes={num_processes} "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} ' + fluxargs + f' --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="{logging_dir}" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --save_every_n_epochs={save_every_n_epochs} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --mixed_precision={mixed_precision} --save_precision={save_precision} --seed={theseed} --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} {cache_args} {raw_args} {test_args} {extargs}'
        print(command)
        # print(extargs)
        subprocess.run(command, shell=True,cwd=sd_script_dir)
        print("Train finished")
        return ()
        
class TensorboardAccess:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
           
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "opentensorboard"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def opentensorboard(self):
        command = f'tensorboard --logdir="{logging_dir}"'
        subprocess.Popen(command, shell=True)
        return()