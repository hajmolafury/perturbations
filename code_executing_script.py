import os
import sys

if sys.argv[1]=='0':
	# GPU1
	os.system("python master.py --learning_rate 1 --alpha 1 --n_hl 1 --hl_size 300")
	os.system("python master.py --learning_rate 3 --alpha 0.8 --n_hl 3 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 0.8 --n_hl 3 --hl_size 300")
	#GPU2
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 0.8 --n_hl 4 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 3 --alpha 0.8 --n_hl 4 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.01 --alpha 0 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 1 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 1 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 0.7 --n_hl 4 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 3 --alpha 0.7 --n_hl 4 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 0.7 --n_hl 5 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 3 --alpha 0.7 --n_hl 5 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 10 --alpha 0.7 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 10 --alpha 0.7 --n_hl 2 --hl_size 300")
	# # GPU3
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 0.8 --n_hl 4 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 1 --alpha 0.8 --n_hl 5 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.08 --alpha 0 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 10 --alpha 0.8 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 20 --alpha 0.8 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.5 --alpha 0.7 --n_hl 8 --hl_size 300")

	# #GPU4
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 0.8 --n_hl 5 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 3 --alpha 0.8 --n_hl 5 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.08 --alpha 0.8 --n_hl 1 --hl_size 5000")


if sys.argv[1]=='1':
	# GPU2
    os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 0.03 --alpha 0 --n_hl 1 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 0.03 --alpha 0 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.9 --n_hl 4 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.9 --n_hl 5 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 10 --alpha 1 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 10 --alpha 1 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.9 --n_hl 8 --hl_size 300")


	# GPU3
	# GPU4
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 3 --alpha 0.8 --n_hl 7 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.8 --n_hl 7 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 0.8 --n_hl 1 --hl_size 10000")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 1 --n_hl 6 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 0.7 --n_hl 7 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 0.5 --alpha 0.7 --n_hl 8 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 0.5 --alpha 0.6 --n_hl 7 --hl_size 300")
    os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 1 --n_hl 2 --hl_size 10")

	# #GPU 5.1
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 0.8 --n_hl 8 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.8 --n_hl 3 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 4 --alpha 0.8 --n_hl 8 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 0.9 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 10 --alpha 0.9 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 10 --alpha 0.9 --n_hl 3 --hl_size 300")


if sys.argv[1]=='2':
	#GPU2
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.8 --n_hl 8 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.8 --n_hl 6 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 0.05 --alpha 0 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.7 --n_hl 8 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.7 --n_hl 8 --hl_size 300")

	#GPU3
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.8 --n_hl 8 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 0.1 --alpha 0 --n_hl 1 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.8 --n_hl 1 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.8 --n_hl 1 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.9 --n_hl 6 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 10 --alpha 1 --n_hl 3 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 10 --alpha 0.9 --n_hl 1 --hl_size 300")


	# #GPU4
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.8 --n_hl 6 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.8 --n_hl 6 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 0.3 --alpha 0 --n_hl 1 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.8 --n_hl 1 --hl_size 10000")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.8 --n_hl 2 --hl_size 300")
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.8 --n_hl 2 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 10 --alpha 0.8 --n_hl 3 --hl_size 300")
    os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.7 --n_hl 1 --hl_size 300")
    os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.7 --n_hl 2 --hl_size 300")


	# GPU5
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 3 --alpha 0.9 --n_hl 8 --hl_size 300")
    # os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 0.9 --n_hl 8 --hl_size 300")
