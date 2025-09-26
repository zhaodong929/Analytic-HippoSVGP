# HIPPOSVGP


## Installation

```bash
# if using anaconda
# conda create -n test_HIPPOSVGP python=3.10 
# conda activate test_HIPPOSVGP
# install in editable mode
python -m pip install -e /path/to/HIPPOSVGP
# https://cds.climate.copernicus.eu/how-to-api follow instructions to install cdsapi and get an api key
```

## Running

1D Solar

```
python -m scripts.onedim.solar.train_full_first2 # run this frist
python -m scripts.onedim.solar.test_vff_online
python -m scripts.onedim.solar.train_svgp_online
python -m scripts.onedim.solar.train_svgp_online_pivini
python -m scripts.onedim.solar.test_hsgp_online_fixedkernel
```

1D COVID 
```
python -m scripts.onedim.covid.test_osvgp_piv
python -m scripts.onedim.covid.test_osvgp
python -m scripts.onedim.covid.test_ohsgp
```

UCI

```
python -m scripts.multidim.gp_ini
python -m scripts.multidim.main_hsgp_online_toy
python -m scripts.multidim.main_hsgp_powerplant
python -m scripts.multidim.main_hsgp_skill
python -m scripts.multidim.main_svgp_online_toy
python -m scripts.multidim.main_svgp_powerplant_piv
python -m scripts.multidim.main_svgp_powerplant
python -m scripts.multidim.main_svgp_skill
```

GPVAE
```
python -m scripts.gpvae.test_osvgpvae_era5 
python -m scripts.gpvae.test_osvgpvae_ovc_era5
python -m scripts.gpvae.test_ohsvgpvae_era5 # run osvgpvae_era5 first before hippo because some hyperparameters for task 1 are initiated the same
```

## Third-party code and licenses

This project includes code adapted from the HiPPO project by HazyResearch.

- Source: [HazyResearch/hippo-code `model/hippo.py` and `model/op.py`](https://github.com/HazyResearch/hippo-code)
- Adapted into: `hipposvgp/hippo.py`
- License: Apache-2.0 (see upstream [LICENSE](https://github.com/HazyResearch/hippo-code/blob/master/LICENSE))

For full third-party license text and details, see `THIRD_PARTY_NOTICES.md` in this repository.