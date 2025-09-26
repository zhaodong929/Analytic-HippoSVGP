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

1D COVID 
```
python -m scripts.onedim.covid.test_osvgp_piv
python -m scripts.onedim.covid.test_osvgp
python -m scripts.onedim.covid.test_ohsgp
```

UCI

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