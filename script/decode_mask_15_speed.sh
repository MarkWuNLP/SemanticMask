export CUDA_VISIBLE_DEVICES=0,1,2,3
source /home/espnet/tools/venv/bin/activate
dataset=test_other
decodedir=/teamscratch/tts_intern_experiment/yuwu1/librispeech_0.5/results/spec_aug_ddp_cnn_mask15_speed_${dataset}_beam50
beam=20
nbest=20
ngpu=4
ctcw=0.5
rnnlmw=0.7
jobperGPU=3
((nblock=$ngpu*$jobperGPU))
echo $nblock
startblock=0
endblock=$nblock

mkdir -p ${decodedir}/data
#--model /teamscratch/tts_intern_experiment/yuwu1/ASR/xiaoying_custom1/train_fb_elayer6_drop0.1_new/results/model.acc.best \
python  /teamscratch/tts_intern_experiment/yuwu1/ASR/utils/split_data.py \
    --parts ${nblock} \
    --json /teamscratch/tts_intern_experiment/yuwu1/librispeech/${dataset}/data_unigram5000.json \
    --datadir ${decodedir}/data
echo "split data to ${nblock} parts done!"


function run(){
    part=$1
cgpu=$[$2+0]
echo $cgpu
cd  /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.5.0_dev/espnet
    export CUDA_VISIBLE_DEVICES=$cgpu
python  /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.5.0_dev/espnet/espnet/bin/asr_recog.py \
	--ngpu 1 \
	--batchsize 0 \
        --result-label ${decodedir}/data${part}.json \
        --backend pytorch \
        --recog-json ${decodedir}/data/data_unigram5000_${part}.json \
        --model /teamscratch/tts_intern_experiment/yuwu1/ASR/librispeech_0.4.0/transformer_ourarch_mask_speed_finetune/results/snapshot.test \
        --beam-size ${beam} \
        --penalty 0.0 \
        --maxlenratio 0.0 \
        --minlenratio 0.0 \
        --ctc-weight ${ctcw} \
        --rnnlm /teamscratch/tts_intern_experiment/yuwu1/ASR/irielm.ep11.last5.avg/rnnlm.model.best \
        --lm-weight ${rnnlmw} \
        --nbest ${nbest} \
        --verbose 1
    }
# /teamscratch/tts_intern_experiment/yuwu1/ASR/script/xiaoying_custom1/lm_test_multi/train_rnnlm_pytorch_2layer_unit1024_adam_bs1024_reversefalse_unigram5000/
# /teamscratch/tts_intern_experiment/yuwu1/ASR/script/xiaoying_custom1/lm_news_single/train_rnnlm_pytorch_2layer_unit1024_adam_bs1024_reversefalse_unigram5000/rnnlm.model.best
#--transformerlm /teamscratch/tts_intern_experiment/yuwu1/ASR/script/xiaoying_custom1/transformer_lm/train_transformerlm_pytorch_4layer_unit768_ahead16_dff1024_adam_bs256_reversefalse_unigram5000/rnnlm.model.best
for ((i=$startblock;i<$endblock;i+=${ngpu}));
do
    for ((j=0;j<${ngpu};j++));
    do
        run $(($i+$j)) $j &
    done
    
    if [ $(( $[$i+1]%${jobperGPU} )) -eq 0 ]
    then
    wait;
    fi
    
    echo "decode $[$i+$ngpu]/${nblock} done"
done
wait
echo "decode all done!"

python /teamscratch/tts_intern_experiment/yuwu1/ASR/utils/merge_data.py \
    --parts ${nblock} \
    --result-dir ${decodedir} \
    --result-label ${decodedir}/data.json
echo "merge result json done!"

python /teamscratch/tts_intern_experiment/yuwu1/ASR/utils/compute_WER.py -i ${decodedir}/data.json
