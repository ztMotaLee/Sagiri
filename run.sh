####Examples of using our LS-Sagiri pipeline.
python infer_LSSagiri.py \
--input your/input/path \
--config configs/model/cldm.yaml \
--ckpt path/to/your/file \
--reload_swinir --swinir_ckpt path/to/your/file \
--steps 30 \
--sr_scale 1 \
--output path/to/your/file \
--device cuda
###Examples of using Sagiri on other restoration models. Note that you also need to change lq image path in infer_Sagiri.py.
python infer_Sagiri.py \
--config configs/model/cldm.yaml \
--ckpt path/to/your/file \
--steps 30 --sr_scale 1 \
--input your/input/path \
--color_fix_type wavelet \
--output path/to/your/file \
--disable_preprocess_model \
--device cuda