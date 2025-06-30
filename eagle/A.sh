CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path /home/ruiyang.chen/hfd/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/ruiyang.chen/hfd/Llama-3.1-8B-Instruct --use_eagle3 > runeval8b_genrate.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.check_llama3_attention --ea-model-path /home/ruiyang.chen/hfd/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/ruiyang.chen/hfd/Llama-3.1-8B-Instruct --use_eagle3 --max-new-toke 120 > runcheckattnv2.log 2>&1 &




