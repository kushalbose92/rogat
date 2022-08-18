# python -u main.py --dataset 'cora' --lr 0.005 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 2000 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.70 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'citeseer' --lr 0.005 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 2000 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.001 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'pubmed' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.001 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'amazonphoto' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'amazoncomputers' --lr 0.02 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.00001 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'coauthorcs' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.001 --device 'cuda:0' | tee output.txt 

 python -u main.py --dataset 'coauthorphysics' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.001 --device 'cpu' | tee output.txt 

# ==========================================================================================================================================================

#  python -u main.py --dataset 'cora' --lr 0.005 --seed 0 --num_layers 20 --hidden_dim 64 --train_iter 500 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

#  python -u main.py --dataset 'cora' --lr 0.005 --seed 0 --num_layers 8 --hidden_dim 64 --train_iter 2000 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.20 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

#  python -u main.py --dataset 'cora' --lr 0.005 --seed 0 --num_layers 32 --hidden_dim 64 --train_iter 2000 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.20 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

# ==========================================================================================================================================================

# python -u main.py --dataset 'amazonphoto' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

#  python -u main.py --dataset 'amazonphoto' --lr 0.01 --seed 0 --num_layers 4 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

#  python -u main.py --dataset 'amazonphoto' --lr 0.01 --seed 0 --num_layers 8 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.60 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'amazonphoto' --lr 0.001 --seed 0 --num_layers 12 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'amazonphoto' --lr 0.001 --seed 0 --num_layers 16 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'amazonphoto' --lr 0.001 --seed 0 --num_layers 20 --hidden_dim 64 --train_iter 1500 --test_iter 1 --use_saved_model True --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.0 --device 'cuda:0' | tee output.txt 

# ==========================================================================================================================================================

# python -u main.py --dataset 'amazoncomputers' --lr 0.02 --seed 0 --num_layers 6 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.30 --w_decay 0.00001 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'cora' --lr 0.005 --seed 0 --num_layers 64 --hidden_dim 64 --train_iter 2000 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.70 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'pubmed' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 500 --test_iter 1 --use_saved_model False --nheads 8 --alpha 0.2 --dropout 0.10 --w_decay 0.001 --device 'cuda:0' | tee output.txt 

# python -u main.py --dataset 'citeseer' --lr 0.005 --seed 0 --num_layers 20 --hidden_dim 64 --train_iter 500 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.20 --w_decay 0.001 --device 'cuda:0' | tee output.txt 