import os
import sys
#learning rate=1
#alpha (hybrid) : fixed at 0.7

if sys.argv[1]=='0':
	# depth=6
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 1 --n_hl 6")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.5 --alpha 0.8 --n_hl 6")

	# depth=7
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 1 --n_hl 7")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.3 --alpha 0.8 --n_hl 7")

	# depth=8
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 5 --alpha 1 --n_hl 8")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=0 python master.py --learning_rate 0.3 --alpha 0.8 --n_hl 8")

if sys.argv[1]=='1':

	# depth=4
	os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 1 --n_hl 4")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 1 --alpha 0.8 --n_hl 4")

	# depth=5
	os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 5 --alpha 1 --n_hl 5")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=1 python master.py --learning_rate 0.5 --alpha 0.8 --n_hl 5")

if sys.argv[1]=='2':

	# depth=1
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 1 --n_hl 1")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.8 --n_hl 1")

	# depth=2
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 1 --n_hl 2")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.8 --n_hl 2")
	#depth=3
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 5 --alpha 1 --n_hl 3")
	# hybrid
	os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.8 --n_hl 3")

	#Alpha=0.7
	# depth=1
	# hybrid
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.7 --n_hl 1")
	# # depth=2
	# # hybrid
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.7 --n_hl 2")
	# #depth=3
	# # hybrid
	# os.system("CUDA_VISIBLE_DEVICES=2 python master.py --learning_rate 1 --alpha 0.7 --n_hl 3")


