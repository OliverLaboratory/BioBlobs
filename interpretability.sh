CUDA_VISIBLE_DEVICES=1 python run_interpretability.py \
    --checkpoint_path /home/wangx86/ICLR-2026/BioBlobs/outputs/bioblobs_multistage/ec/structure/2025-11-26-21-42-37/stage1/best-stage1-epoch=13-val_acc=0.652.ckpt \
    --dataset_name ec \
    --split structure \
    --split_type test \
    --batch_size 4 \
    --num_workers 4 \
    --max_batches 1 \
    --output_dir ./interpretability_results \
    --data_dir /data/oliver_lab/wangx86/bioblobs/proteinshake_data \

