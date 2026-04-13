exec > >(tee -a "run.log") 2>&1

export CUDA_VISIBLE_DEVICES=0

# Alpha ablation (fixed fc_start_step=3, fc_interval=6)
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.1 --output_dir samples/alpha0.1
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.3 --output_dir samples/alpha0.3
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.5 --output_dir samples/alpha0.5
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.7 --output_dir samples/alpha0.7
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.9 --output_dir samples/alpha0.9

# Interval ablation (fixed fc_start_step=3, fc_alpha=0.3)
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 2 --fc_alpha 0.3 --output_dir samples/N2
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 3 --fc_alpha 0.3 --output_dir samples/N3
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 4 --fc_alpha 0.3 --output_dir samples/N4
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 5 --fc_alpha 0.3 --output_dir samples/N5
torchrun --nproc_per_node=1 sample.py --fc_start_step 3 --fc_interval 6 --fc_alpha 0.3 --output_dir samples/N6

export CUDA_VISIBLE_DEVICES=0
python evaluate.py --test_folder samples/alpha0.1
echo "-------------alpha0.1-------------"
python evaluate.py --test_folder samples/alpha0.3
echo "-------------alpha0.3-------------"
python evaluate.py --test_folder samples/alpha0.5
echo "-------------alpha0.5-------------"
python evaluate.py --test_folder samples/alpha0.7
echo "-------------alpha0.7-------------"
python evaluate.py --test_folder samples/alpha0.9
echo "-------------alpha0.9-------------"
python evaluate.py --test_folder samples/N2
echo "-------------N2-------------"
python evaluate.py --test_folder samples/N3
echo "-------------N3-------------"
python evaluate.py --test_folder samples/N4
echo "-------------N4-------------"
python evaluate.py --test_folder samples/N5
echo "-------------N5-------------"
python evaluate.py --test_folder samples/N6
echo "-------------N6-------------"
