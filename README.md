# *Lasagna:* Layered Score Distillation for Disentangled Object Relighting
**[Dina Bashkirova](https://cs-people.bu.edu/dbash/), [Arijit Ray](https://arijitray1993.github.io/), [Rupayan Mallick](https://contact.georgetown.edu/view/rm2083/),
[Sarah Adel Bargal](https://bargal.georgetown.domains/), [Jianming Zhang](https://jimmie33.github.io/), [Ranjay Krishna](https://ranjaykrishna.com/index.html), [Kate Seanko](http://ai.bu.edu/ksaenko.html/)** </br>
[arxiv](https://arxiv.org/pdf/2312.00833.pdf) | [ReLiT dataset (coming soon)]()</br>
We propose Lasagna, a layered image editing approach that allows controlled and language-guided object relighting. Lasagna achieves a controlled relighting via layered score distillation sampling that allows extracting the diffusion model lighting prior without changing other crucial aspects of the input image.
<!-- ![img](https://cs-people.bu.edu/dbash/img/i2i_eval.png) -->

<p align="center">
  <img src="https://cs-people.bu.edu/dbash/img/lasagna.png" />
</p>

## Setup
### Install dependencies
````
conda create -y -n lasagna python=3.10.11
conda activate lasagna
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install accelerate
pip install git+https://github.com/huggingface/diffusers

````

### Download the ControlNet checkpoint
```
wget https://cs-people.bu.edu/dbash/checkpoints/relit_controlnet.zip
unzip relit_controlnet.zip ./
```
## Training Lasagna to relight an input image
```

```
