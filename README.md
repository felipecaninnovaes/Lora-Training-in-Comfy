**Título:** Treino de lora no ComfyUI

**Descrição:**

Este módulo permite a execução do [SD-Scripts](https://github.com/kohya-ss/sd-scripts) através da interface do ComfyUI, tornando mais fácil e conveniente a execução dos comandos. Agora você pode criar e executar scripts sem precisar sair da interface do Comfy!

**Características:**

* Support a (SD1.5, SD2.0, SD2.1, SDXL, Flux1.0(Experimental ainda)).
* Support a Multi-GPU
* Flexibilidade e facilidade para configura sem precisa mexer com linhas no terminal.

**Sistemas suportados:**
Este modulo foi testado apenas em sistemas que usam como base o kernel do Linux. Windows e MacOS não foram testados.

**Como usar:**

1. Instale o módulo no seu projeto ComfyUI.
  ``` bash
  cd CAMINHO_DO_SEU_COMFYUI/custom_nodes
  git clone https://github.com/felipecaninnovaes/Lora-Training-in-Comfy
  cd Lora-Training-in-Comfy
  python -m pip install -r requirements.txt
  ```
2. Importe os presets disponiveis na pasta presets no modulo ou crie apartir da interface do ComfyUI

**Requisitos:**
* **Softwares**
  * ComfyUI mais atualizado
  * Python 3.10.x
  * Nvidia cuda 12.4+
* **Hardware**
  * Nvidia GPU 6Gb ou mais (para usar flux acima de 12GB)
  * Memoria RAM 16Gb (para usar flux no minimo 32Gb e 20Gb de Swap, Recomendado 48GB de RAM ou mais)

**Hardware Testado**

* CPU: [Ryzen 7 5700x](https://www.amd.com/pt/product/11826) 
* Placa Mae: [GA-AB350M-DS3H V2](https://www.gigabyte.com/br/Motherboard/GA-AB350M-DS3H-V2-rev-11#kf) 
* Memoria RAM: 32Gb 
* Memoria Swap: 32Gb 
* GPU: RTX 3060 12Gb VRAM (GPU Primaria), GTX 1660 Super 6Gb VRAM (Segunda GPU) 
* SSD: NVME 1TB 

|MODELO|FUNCINA?|VELOCIDADE|
|------|--------|----------
|SD1.5|SIM|EXELENTE|
|SD2.0|SIM|EXELENTE|
|SD2.1|SIM|EXELENTE|
|SDXL|SIM|BOM|
|FLUX1.0|SIM|LENTO|

**Documentação:** 

Em desenvolvimento.