# CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/ruiyang.chen/hfd/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/ruiyang.chen/hfd/Llama-3.1-8B-Instruct --use_eagle3 > prune.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path /home/ruiyang.chen/hfd/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/ruiyang.chen/hfd/Llama-3.1-8B-Instruct --use_eagle3 > base.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 > runnormal.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --bench-name gsm8k > runnormalgsm8k.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric topk --important_metric_value 100 > prune1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric toppercent --important_metric_value 0.25 > prunepct025.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric topp --important_metric_value 0.25 > prunetopp025.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop150p --important_metric_value 0.25 > prunemaxtop150p025.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric topp --important_metric_value 0.4 > prunetopp04.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.15 > prunemaxtop100p015.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.15 --bench-name qa > prunemaxtop100p015.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.15 --bench-name qa --total-token 50 > prunemaxtop100p015.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.check_llama3_attention --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --bench-name qa > runcheckattn.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop100p --important_metric_value 0.15 --bench-name qa > prunemaxtop100p015.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop50p --important_metric_value 0.05  > prunemaxtop50p005.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_prune --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --important_metric maxtop75p --important_metric_value 0.05  > prunemaxtop75p005.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 6 --abit 6 > runquant66.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 > runquant8.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 4 > runquant84.log 2>&1 &

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 6 --abit 6 > runquant6.log 2>&1 &

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 6 --abit 6 > runquant6.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 6 > runquant8886.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_multiquant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --wbit_low 8 --abit_low 6 --bench-name gsm8k > runquant8886gsm8k.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -m eagle.evaluation.gen_ea_answer_llama3chat_quant --ea-model-path /home/chenruiyang/Code/LLM/HFD/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path /home/chenruiyang/Code/LLM/HFD/Llama-3.1-8B-Instruct --use_eagle3 --mode ant-int --wbit 8 --abit 8 --bench-name gsm8k > rungsm8kquant88.log 2>&1 &


