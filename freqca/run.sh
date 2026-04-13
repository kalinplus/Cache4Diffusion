exec > >(tee -a "run.log") 2>&1

export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method None --output_dir samples/N3_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method None --output_dir samples/N4_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method None --output_dir samples/N5_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method None --output_dir samples/N6_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method None --output_dir samples/N7_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method None --output_dir samples/N8_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method None --output_dir samples/N9_None_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method None --output_dir samples/N10_None_NoZCache

torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method FFT --output_dir samples/N3_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method FFT --output_dir samples/N4_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method FFT --output_dir samples/N5_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method FFT --output_dir samples/N6_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method FFT --output_dir samples/N7_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method FFT --output_dir samples/N8_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method FFT --output_dir samples/N9_FFT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method FFT --output_dir samples/N10_FFT_NoZCache

torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method DCT --output_dir samples/N3_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method DCT --output_dir samples/N4_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method DCT --output_dir samples/N5_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method DCT --output_dir samples/N6_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method DCT --output_dir samples/N7_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method DCT --output_dir samples/N8_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method DCT --output_dir samples/N9_DCT_NoZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method DCT --output_dir samples/N10_DCT_NoZCache

torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method None --use_z_cache --forecast_steps 3 --output_dir samples/N3_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method None --use_z_cache --forecast_steps 4 --output_dir samples/N4_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method None --use_z_cache --forecast_steps 5 --output_dir samples/N5_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method None --use_z_cache --forecast_steps 6 --output_dir samples/N6_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method None --use_z_cache --forecast_steps 7 --output_dir samples/N7_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method None --use_z_cache --forecast_steps 8 --output_dir samples/N8_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method None --use_z_cache --forecast_steps 9 --output_dir samples/N9_None_ZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method None --use_z_cache --forecast_steps 10 --output_dir samples/N10_None_ZCache

torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method FFT --use_z_cache --forecast_steps 3 --output_dir samples/N3_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method FFT --use_z_cache --forecast_steps 4 --output_dir samples/N4_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method FFT --use_z_cache --forecast_steps 5 --output_dir samples/N5_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method FFT --use_z_cache --forecast_steps 6 --output_dir samples/N6_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method FFT --use_z_cache --forecast_steps 7 --output_dir samples/N7_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method FFT --use_z_cache --forecast_steps 8 --output_dir samples/N8_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method FFT --use_z_cache --forecast_steps 9 --output_dir samples/N9_FFT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method FFT --use_z_cache --forecast_steps 10 --output_dir samples/N10_FFT_ZCache

torchrun --nproc_per_node=1 sample.py --interval 3 --decompose_method DCT --use_z_cache --forecast_steps 3 --output_dir samples/N3_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 4 --decompose_method DCT --use_z_cache --forecast_steps 4 --output_dir samples/N4_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 5 --decompose_method DCT --use_z_cache --forecast_steps 5 --output_dir samples/N5_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 6 --decompose_method DCT --use_z_cache --forecast_steps 6 --output_dir samples/N6_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method DCT --use_z_cache --forecast_steps 7 --output_dir samples/N7_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 8 --decompose_method DCT --use_z_cache --forecast_steps 8 --output_dir samples/N8_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 9 --decompose_method DCT --use_z_cache --forecast_steps 9 --output_dir samples/N9_DCT_ZCache
torchrun --nproc_per_node=1 sample.py --interval 10 --decompose_method DCT --use_z_cache --forecast_steps 10 --output_dir samples/N10_DCT_ZCache

export CUDA_VISIBLE_DEVICES=0 
python evaluate.py --test_folder samples/N3_None_NoZCache
echo "-------------N3_None_NoZCache-------------"
python evaluate.py --test_folder samples/N4_None_NoZCache
echo "-------------N4_None_NoZCache-------------"
python evaluate.py --test_folder samples/N5_None_NoZCache
echo "-------------N5_None_NoZCache-------------"
python evaluate.py --test_folder samples/N6_None_NoZCache
echo "-------------N6_None_NoZCache-------------"
python evaluate.py --test_folder samples/N7_None_NoZCache
echo "-------------N7_None_NoZCache-------------"
python evaluate.py --test_folder samples/N8_None_NoZCache
echo "-------------N8_None_NoZCache-------------"
python evaluate.py --test_folder samples/N9_None_NoZCache
echo "-------------N9_None_NoZCache-------------"
python evaluate.py --test_folder samples/N10_None_NoZCache
echo "-------------N10_None_NoZCache-------------"

python evaluate.py --test_folder samples/N3_FFT_NoZCache
echo "-------------N3_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N4_FFT_NoZCache
echo "-------------N4_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N5_FFT_NoZCache
echo "-------------N5_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N6_FFT_NoZCache
echo "-------------N6_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N7_FFT_NoZCache
echo "-------------N7_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N8_FFT_NoZCache
echo "-------------N8_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N9_FFT_NoZCache
echo "-------------N9_FFT_NoZCache-------------"
python evaluate.py --test_folder samples/N10_FFT_NoZCache
echo "-------------N10_FFT_NoZCache-------------"

python evaluate.py --test_folder samples/N3_DCT_NoZCache
echo "-------------N3_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N4_DCT_NoZCache
echo "-------------N4_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N5_DCT_NoZCache
echo "-------------N5_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N6_DCT_NoZCache
echo "-------------N6_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N7_DCT_NoZCache
echo "-------------N7_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N8_DCT_NoZCache
echo "-------------N8_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N9_DCT_NoZCache
echo "-------------N9_DCT_NoZCache-------------"
python evaluate.py --test_folder samples/N10_DCT_NoZCache
echo "-------------N10_DCT_NoZCache-------------"

python evaluate.py --test_folder samples/N3_None_ZCache
echo "-------------N3_None_ZCache-------------"
python evaluate.py --test_folder samples/N4_None_ZCache
echo "-------------N4_None_ZCache-------------"
python evaluate.py --test_folder samples/N5_None_ZCache
echo "-------------N5_None_ZCache-------------"
python evaluate.py --test_folder samples/N6_None_ZCache
echo "-------------N6_None_ZCache-------------"
python evaluate.py --test_folder samples/N7_None_ZCache
echo "-------------N7_None_ZCache-------------"
python evaluate.py --test_folder samples/N8_None_ZCache
echo "-------------N8_None_ZCache-------------"
python evaluate.py --test_folder samples/N9_None_ZCache
echo "-------------N9_None_ZCache-------------"
python evaluate.py --test_folder samples/N10_None_ZCache
echo "-------------N10_None_ZCache-------------"

python evaluate.py --test_folder samples/N3_FFT_ZCache
echo "-------------N3_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N4_FFT_ZCache
echo "-------------N4_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N5_FFT_ZCache
echo "-------------N5_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N6_FFT_ZCache
echo "-------------N6_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N7_FFT_ZCache
echo "-------------N7_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N8_FFT_ZCache
echo "-------------N8_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N9_FFT_ZCache
echo "-------------N9_FFT_ZCache-------------"
python evaluate.py --test_folder samples/N10_FFT_ZCache
echo "-------------N10_FFT_ZCache-------------"

python evaluate.py --test_folder samples/N3_DCT_ZCache
echo "-------------N3_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N4_DCT_ZCache
echo "-------------N4_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N5_DCT_ZCache
echo "-------------N5_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N6_DCT_ZCache
echo "-------------N6_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N7_DCT_ZCache
echo "-------------N7_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N8_DCT_ZCache
echo "-------------N8_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N9_DCT_ZCache
echo "-------------N9_DCT_ZCache-------------"
python evaluate.py --test_folder samples/N10_DCT_ZCache
echo "-------------N10_DCT_ZCache-------------"