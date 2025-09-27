# TCFormer: Temporal Convolutional Transformer

> Official code for the paper **“Temporal convolutional transformer for EEG based motor imagery decoding”**  
> Paper: https://www.nature.com/articles/s41598-025-16219-7  (Nature Scientific Reports, 2025).  

- Built upon ideas/code from **EEG‑ATCNet**: https://github.com/Altaheri/EEG-ATCNet  
- Training pipeline structure and certain implementations adapted from **channel‑attention**: https://github.com/martinwimpff/channel-attention
---

<img width="6225" height="3488" alt="ATCFormer_blocks 5 4" src="https://github.com/user-attachments/assets/1ff17b1d-0d81-4f55-b321-9fb13a27df16" />  

TCFormer fuses a **Multi‑Kernel CNN (MK‑CNN)** front‑end, a **Transformer encoder** with **Grouped‑Query Attention (GQA)** + **RoPE**, and a **Temporal Convolutional Network (TCN)** head. The model captures **local** (CNN), **global** (Transformer), and **long‑range** (TCN) temporal dependencies in MI‑EEG.


## Quickstart

### 1) Environment
- Python: **3.10** (tested)
- PyTorch: **2.6.0** (tested)
- CUDA: **12.4** (tested)

Install dependencies from **requirements.txt**:

```bash
pip install -r requirements.txt
```

> Tested on Ubuntu 24.04 with RTX A6000 GPUs (48 GB). Results may vary slightly by hardware and seeds.


## Training & Evaluation

Examples:

```bash
# BCI IV‑2a, within‑subject, with augmentation
python train_pipeline.py --model tcformer --dataset bcic2a --interaug

# BCI IV‑2b, within‑subject, no augmentation
python train_pipeline.py --model tcformer --dataset bcic2b --no_interaug

# HGD, cross‑subject (LOSO), no augmentation
python train_pipeline.py --model tcformer --dataset hgd --loso --no_interaug

```

Batch a full sweep:
```bash
bash run_all.sh
```

Summaries (tables produced under the output dir):
```bash
# Per‑subject (Per‑subject and per seeds)
python summarize_per_subject.py /results/

# Dataset‑level aggregation (per seeds, averages across subjects) 
python summarize_results.py /results/TCFormer/2a
```


### 2) Datasets

| Dataset         | Tasks (classes)                   | Channels        | SR (Hz) | Split (sessions)         | Notes                |
|-----------------|-----------------------------------|-----------------|---------|--------------------------|----------------------|
| [BCI Comp IV‑2a](http://www.bbci.de/competition/iv/)  | L/R hand, Feet, Tongue (4)        | 22 EEG | 250     | S1 train, S2 test        | Motor **imagery**    |
| [BCI Comp IV‑2b](http://www.bbci.de/competition/iv/)  | L vs R hand (2)                   | 3 (C3, Cz, C4)  | 250     | S1–S3 train, S4–S5 test  | Motor **imagery**    |
| [HGD (High‑Gamma)](https://github.com/robintibor/high-gamma-dataset)| L/R hand, Feet, Rest (4)          | 128 → **44**    | 512→**250** | S1 train, S2 test    | Motor **execution**  |
> All above datasets will be downloaded automatically.  
> This reposotory also suport [BCI Comp III‑IVa](https://www.bbci.de/competition/iii/#data_set_iva) and [REH-MI](https://dx.doi.org/10.21227/xgzb-6s98) datasets. For these two datasets, you need to download them and put the files in the directory defined in [load_bcic3.py](https://github.com/Altaheri/TCFormer/blob/main/utils/load_bcic3.py) and [load_reh_mi.py](https://github.com/Altaheri/TCFormer/blob/main/utils/load_reh_mi.py).  


## Results (from the Paper)

Accuracy Summary (Subject-Dependent vs. LOSO, ± Augmentation)

This table reports **mean accuracy (%)** for all models across **BCI IV-2a**, **BCI IV-2b**, and **HGD** in both **subject-dependent (Sub-Dep)** and **LOSO** settings, **with (+aug)** and **without (–aug)** augmentation, plus model **parameter counts (k)**.Parameter counts are referenced from the IV-2a configuration and may vary slightly by dataset.

<table>
  <thead>
    <tr>
      <th rowspan="3">Model</th>
      <th rowspan="3">Params (k)</th>
      <th colspan="4"><a href="https://www.bbci.de/competition/iv/#dataset2a">BCI Comp IV‑2a</a></th>
      <th colspan="4"><a href="https://www.bbci.de/competition/iv/#dataset2b">BCI Comp IV‑2b</a></th>
      <th colspan="4"><a href="https://github.com/robintibor/high-gamma-dataset">HGD (High‑Gamma)</a></th>
    </tr>
    <tr>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
    </tr>
    <tr>
      <th>–aug</th><th>+aug</th>
      <th>–aug</th><th>+aug</th>
      <th>–aug</th><th>+aug</th>
      <th>–aug</th><th>+aug</th>
      <th>–aug</th><th>+aug</th>
      <th>–aug</th><th>+aug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EEGNet</td>
      <td align="left">1.7</td>
      <td align="right">70.39</td><td align="left">72.62</td>
      <td align="right">52.01</td><td align="left">52.03</td>
      <td align="right">82.80</td><td align="left">83.65</td>
      <td align="right">77.67</td><td align="left">77.89</td>
      <td align="right">85.59</td><td align="left">85.94</td>
      <td align="right">57.95</td><td align="left">60.12</td>
    </tr>
    <tr>
      <td>ShallowNet</td>
      <td align="left">44.6</td>
      <td align="right">60.50</td><td align="left">65.72</td>
      <td align="right">48.83</td><td align="left">47.31</td>
      <td align="right">79.12</td><td align="left">81.45</td>
      <td align="right">74.50</td><td align="left">75.58</td>
      <td align="right">89.75</td><td align="left">91.54</td>
      <td align="right"><strong>72.47</strong></td><td align="center">—</td>
    </tr>
    <tr>
      <td>BaseNet</td>
      <td align="left">3.7</td>
      <td align="right">76.45</td><td align="left">78.58</td>
      <td align="right">57.82</td><td align="left">56.89</td>
      <td align="right">84.51</td><td align="left">86.11</td>
      <td align="right">78.55</td><td align="left">78.61</td>
      <td align="right">93.64</td><td align="left">95.40</td>
      <td align="right">68.55</td><td align="center">—</td>
    </tr>
    <tr>
      <td>EEGTCNet</td>
      <td align="left">4.1</td>
      <td align="right">75.62</td><td align="left">78.82</td>
      <td align="right">55.09</td><td align="left">55.99</td>
      <td align="right">85.54</td><td align="left">86.74</td>
      <td align="right">78.82</td><td align="left">80.56</td>
      <td align="right">91.83</td><td align="left">93.54</td>
      <td align="right">60.59</td><td align="center">—</td>
    </tr>
    <tr>
      <td>TS-SEFFNet</td>
      <td align="left">334.8</td>
      <td align="right">76.65</td><td align="center">—</td>
      <td align="right">56.74</td><td align="center">—</td>
      <td align="right">84.18</td><td align="center">—</td>
      <td align="right">77.82</td><td align="center">—</td>
      <td align="right">92.45</td><td align="center">—</td>
      <td align="right">69.99</td><td align="center">—</td>
    </tr>
    <!-- CTNet split into two rows -->
    <tr>
      <td rowspan="2">CTNet,&nbsp;&nbsp;F1=20<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;F1=8</td>
      <td rowspan="2" align="left">152.7<br/>27.3</td>
      <!-- conf-1 -->
      <td align="right">78.08</td><td align="left">81.91</td>
      <td align="right">59.67</td><td align="left">60.09</td>
      <td align="right">86.81</td><td align="left">86.91</td>
      <td align="right">79.44</td><td align="left">80.29</td>
      <td align="right">93.53</td><td align="left">94.21</td>
      <td align="right">64.87</td><td align="left">64.60</td>
    </tr>
    <tr>
      <!-- conf-2: fill in values if/when available -->
      <td align="center">—</td><td align="center">79.24</td>
      <td align="center">—</td><td align="center">56.17</td>
      <td align="center">—</td><td align="center">87.50</td>
      <td align="center">—</td><td align="center">80.15</td>
      <td align="center">—</td><td align="center">92.22</td>
      <td align="center">—</td><td align="center">—</td>
    </tr>
    <tr>
      <td>MSCFormer</td>
      <td align="left">150.7</td>
      <td align="right">75.25</td><td align="left">79.16</td>
      <td align="right">52.04</td><td align="left">54.27</td>
      <td align="right">85.57</td><td align="left">87.60</td>
      <td align="right">78.88</td><td align="left">79.20</td>
      <td align="right">91.33</td><td align="left">94.31</td>
      <td align="right">61.06</td><td align="left">61.19</td>
    </tr>
    <tr>
      <td>EEGConformer</td>
      <td align="left">789.6</td>
      <td align="right">70.70</td><td align="left">75.39</td>
      <td align="right">45.44</td><td align="left">45.59</td>
      <td align="right">79.46</td><td align="left">81.89</td>
      <td align="right">73.44</td><td align="left">75.25</td>
      <td align="right">93.60</td><td align="left">94.67</td>
      <td align="right">69.21</td><td align="left">69.92</td>
    </tr>
    <tr>
      <td>ATCNet</td>
      <td align="left">113.7</td>
      <td align="right"><strong>83.40</strong></td><td align="left">83.78</td>
      <td align="right">60.05</td><td align="left">59.66</td>
      <td align="right">86.25</td><td align="left">86.26</td>
      <td align="right"><strong>80.29</strong></td><td align="left">80.94</td>
      <td align="right">93.65</td><td align="left">95.08</td>
      <td align="right">67.42</td><td align="left">—</td>
    </tr>
    <tr>
      <td><strong>TCFormer (proposed)</strong></td>
      <td align="left">77.8</td>
      <td align="right">83.06</td><td align="left"><strong>84.79</strong></td>
      <td align="right"><strong>62.44</strong></td><td align="left"><strong>63.00</strong></td>
      <td align="right"><strong>87.11</strong></td><td align="left"><strong>87.71</strong></td>
      <td align="right">79.73</td><td align="left"><strong>81.34</strong></td>
      <td align="right"><strong>95.62</strong></td><td align="left"><strong>96.27</strong></td>
      <td align="right">71.90<sup>1</sup></td><td align="left"><strong>72.83<sup>1</sup></strong></td>
    </tr>
  </tbody>
</table>

<sup>1</sup> Using a deeper TCFormer encoder (**N = 5**, ≈131 k params), refer to the [paper](https://www.nature.com/articles/s41598-025-16219-7) for more details.

![Figure 8 1](https://github.com/user-attachments/assets/292066bd-e0e8-4586-bbf1-e466b2f2ba97)

---

## Citation

Please cite the paper if you use this code:

```bibtex
@article{Altaheri2025TCFormer,
  title   = {Temporal convolutional transformer for EEG based motor imagery decoding},
  author  = {Altaheri, Hamdi and Karray, Fakhri and Karimi, Amir-Hossein},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-16219-7}
}
```

---

## Acknowledgements & License

- Built upon ideas/code from **EEG‑ATCNet**: https://github.com/Altaheri/EEG-ATCNet  
- Training pipeline structure and certain implementations adapted from **channel‑attention**: https://github.com/martinwimpff/channel-attention

This repository is released under the **MIT License** (see `LICENSE`).

Contact: Hamdi Altaheri
