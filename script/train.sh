export CUDA_VISIBLE_DEVICES=0,1,2,3
source /home/espnet/tools/venv/bin/activate
expdir=/teamscratch/tts_intern_experiment/yuwu1/ASR/librispeech_0.4.0/transformer_release_model
mkdir -p ${expdir}
cd  /teamscratch/tts_intern_experiment/yuwu1/Azure_Code/ASR_SemanticMask
python espnet/bin/asr_train.py \
        --config  /teamscratch/tts_intern_experiment/yuwu1/Azure_Code/ASR_SemanticMask/configs/train_pytorch_transformer_large_ngpu4.yaml  \
	--preprocess-conf   /teamscratch/tts_intern_experiment/yuwu1/Azure_Code/ASR_SemanticMask/configs/specaug.yaml \
        --ngpu 4 \
	--dist true \
	--maskwords true \
        --backend pytorch \
        --dict /teamscratch/tts_intern_experiment/yuwu1/ASR/script/train_960_unigram5000_units.txt \
        --train-json /teamscratch/tts_intern_experiment/yuwu1/librispeech/train_960/data_aligned_clean_sp_new.json \
        --valid-json  /teamscratch/tts_intern_experiment/yuwu1/librispeech/dev_clean/data_unigram5000.json  \
        --outdir ${expdir}/results \
        > ${expdir}/train.log
