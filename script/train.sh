export CUDA_VISIBLE_DEVICES=0,1,2,3
#source /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.4.0/espnet/espnet/tools/venv/bin/activate
source /home/espnet/tools/venv/bin/activate
#bash /home/espnet/egs/librispeech/asr1/path.sh
#expdir=/home/yuwu1/tts_exp
expdir=/teamscratch/tts_intern_experiment/yuwu1/ASR/librispeech_0.4.0/transformer_ourarch_mask_speed
#echo Start
mkdir -p ${expdir}
#cp -r $(cd `dirname $0`; pwd)/../../ ${expdir}/E2ESR; chmod 755 ${expdir}/E2ESR
#echo Start
cd  /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.5.0_dev/espnet
python espnet/bin/asr_train.py \
        --config /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.4.0/config/train_pytorch_transformer_large_ngpu4.yaml  \
	--preprocess-conf  /teamscratch/tts_intern_experiment/yuwu1/ESPnet_0.4.0/espnet/espnet/bin/conf/specaug.yaml \
        --ngpu 4 \
	--dist true \
	--maskwords true \
	--use_torchspeech false \
        --backend pytorch \
        --dict /teamscratch/tts_intern_experiment/yuwu1/ASR/script/train_960_unigram5000_units.txt \
        --train-json /teamscratch/tts_intern_experiment/yuwu1/librispeech/train_960/data_aligned_clean_sp.json \
        --valid-json  /teamscratch/tts_intern_experiment/yuwu1/librispeech/dev_clean/data_unigram5000.json  \
        --outdir ${expdir}/results \
        > ${expdir}/train.log
