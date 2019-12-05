def get_n_params(model):
    pp=0
    for p in list(model[0].parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

from espnet.asr.pytorch_backend.asr_init import load_trained_model
model = load_trained_model(r"/teamscratch/tts_intern_experiment/yuwu1/ASR/librispeech_0.4.0/transformer_ourarch_mask_speed_finetune/results/snapshot.ep.60")
get_n_params(model)