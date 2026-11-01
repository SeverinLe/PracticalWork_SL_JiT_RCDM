---
tags:
- pytorch
extra_gated_fields:
  First Name: text
  Last Name: text
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Other
  geo: ip_location
extra_gated_button_content: Submit
---
# Model Card for RETFound_DINOv3

<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Yukun Zhou
- **Model type:** Pre-trained model
- **License:** Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [RETFound](https://github.com/HORIZONHealthcare/Med_DINOv3)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

This repo contains the model weight. After granted the access, please fill the token in the [code](https://github.com/HORIZONHealthcare/Med_DINOv3).

The code will automatically download the model and run the training. 



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->


- **Hardware Type:** 4 * NVIDIA A100 80GB
- **Hours used:** 10 days
- **Cloud Provider:** UCL CS Cluster


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

```
@article{zhou2026understanding,
  title={Understanding pre-training data effects in retinal foundation models using two large fundus cohorts},
  author={Zhou, Yukun and Wang, Zheyuan and Wu, Yilan and Ong, Ariel Yuhan and Wagner, Siegfried K and Ruffell, Eden and Chia, Mark A and Guan, Zhouyu and Ju, Lie and Engelmann, Justin and others},
  journal={Nature Communications},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

```
@misc{zhou2025generalistversusspecialistvision,
      title={Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics}, 
      author={Yukun Zhou and Paul Nderitu and Jocelyn Hui Lin Goh and Justin Engelmann and Siegfried K. Wagner and Anran Ran and Hongyang Jiang and Lie Ju and Ke Zou and Sahana Srinivasan and Hyunmin Kim and Takahiro Ninomiya and Zheyuan Wang and Gabriel Dawei Yang and Eden Ruffell and Dominic Williamson and Rui Santos and Gabor Mark Somfai and Carol Y. Cheung and Tien Yin Wong and Daniel C. Alexander and Yih Chung Tham and Pearse A. Keane},
      year={2025},
      eprint={2509.03421},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.03421}, 
}
```

## Model Card Contact

**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk**
