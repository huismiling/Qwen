
mdl_path=$PWD/output_qwen_0403/checkpoint-2800
data_path=$PWD/data/total_devset.txt

python run_gptq.py \
    --model_name_or_path $mdl_path \
    --data_path $data_path \
    --group-size -1 \
    --out_path $mdl_path-gptq-1

