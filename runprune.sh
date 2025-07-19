
# CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --bench-name gsm8k > runnormalgsm8k.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --bench-name gsm8k > rungsm8kquant88.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 4 --bench-name gsm8k > runquant8884gsm8k.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 4 --abit 4 --bench-name gsm8k > rungsm8kquant44.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 4 --bench-name gsm8k > rungsm8kquant84.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 6 --bench-name gsm8k > rungsm8kquant86.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 6 --bench-name gsm8k > runquant8886gsm8k.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 6 --abit 6 --bench-name gsm8k > rungsm8kquant66.log 2>&1 ;

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 16 --abit 16 --wbit_low 16 --abit_low 8 --question_nums 5 --no_outlier > runquant1616168.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 7 --bench-name gsm8k --question_nums 20 > runquant8887gsm8k.log 2>&1 ;

# CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 5 --bench-name gsm8k --question_nums 20 > runquant8885gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop150p --important_metric_value 0.25 --bench-name gsm8k --question_nums 20 > prunemaxtop150p025gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop150p --important_metric_value 0.15 --bench-name gsm8k --question_nums 20 > prunemaxtop150p015gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.25 --bench-name gsm8k --question_nums 20 > prunemaxtop100p025gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.15 --bench-name gsm8k --question_nums 20 > prunemaxtop100p015gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop50p --important_metric_value 0.25 --bench-name gsm8k --question_nums 20 > prunemaxtop50p025gsm8k.log 2>&1 ;

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop50p --important_metric_value 0.15 --bench-name gsm8k --question_nums 20 > prunemaxtop50p015gsm8k.log 2>&1 ;