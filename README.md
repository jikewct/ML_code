# ML_code
## grad_checkpoint 
 -afhq cat : uvit (32, 128,128)   patch_size=8, embed_dim=768,depth=16, num_heads=12,
| mixed precision training | xformers | gradient checkpointing |  training speed (second/steps)  |    memory     |
|:------------------------:|:--------:|:----------------------:|:-----------------:|:-------------:|
|            ❌            |    ❌     |           ❌            |         -        | out of memory |
|            ✔             |    ❌     |           ❌            | 4.34            |   7352 MB    |
|            ✔             |    ✔     |           ❌            | 4.36             |   7388 MB    |
|            ✔             |    ✔     |           ✔            | 5.52             |   4048 MB    |

xformers 在 gtx1080上无效 https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5581