exec > >(tee -a "run.log") 2>&1

export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 sample.py --interval 3 --output_dir samples/N3
torchrun --nproc_per_node=1 sample.py --interval 4 --output_dir samples/N4
torchrun --nproc_per_node=1 sample.py --interval 5 --output_dir samples/N5
torchrun --nproc_per_node=1 sample.py --interval 6 --output_dir samples/N6
torchrun --nproc_per_node=1 sample.py --interval 7 --output_dir samples/N7
torchrun --nproc_per_node=1 sample.py --interval 8 --output_dir samples/N8
torchrun --nproc_per_node=1 sample.py --interval 9 --output_dir samples/N9
torchrun --nproc_per_node=1 sample.py --interval 10 --output_dir samples/N10

export CUDA_VISIBLE_DEVICES=0 
python evaluate.py --test_folder samples/N3
echo "-------------N3-------------"
python evaluate.py --test_folder samples/N4
echo "-------------N4-------------"
python evaluate.py --test_folder samples/N5
echo "-------------N5-------------"
python evaluate.py --test_folder samples/N6
echo "-------------N6-------------"
python evaluate.py --test_folder samples/N7
echo "-------------N7-------------"
python evaluate.py --test_folder samples/N8
echo "-------------N8-------------"
python evaluate.py --test_folder samples/N9
echo "-------------N9-------------"
python evaluate.py --test_folder samples/N10
echo "-------------N10-------------"
