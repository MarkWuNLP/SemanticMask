
import json, editdistance

from operator import itemgetter
def compute_wer_txt(path):


    for count in range(50):
        f = open(path)
        err, cnt = 0, 0
        #print("Start Computing WER")
        d = {}
        for line in f:
            if len(line.strip().split('\t')) != 5:
                continue
            hyp, ref, rec_token, lmscore, amscore = line.strip().split("\t")
            lmscore = float(lmscore)
            amscore = float(amscore)
            if ref not in d:
                d[ref] = []
           # amscore /= len(rec_token.split())
           # amscore =- amscore
            #print(lmscore, amscore)
            lmscore = lmscore * len(rec_token.split())
            d[ref].append([hyp, ref, rec_token,  +(0*0.02) * float(lmscore) -
                           1 * float(amscore) - (0.1*count + 0.5) * len(rec_token.split())])

        for key in d:
            res = sorted(d[key], key=itemgetter(-1))
            predict = res[0][0]
            ref = res[0][1]
            #print(ref)

            ref_token = ref.strip().split()
            predict_token = predict.strip().split()

            e = editdistance.eval(ref_token,predict_token)
            #print(ref_token,predict_token,e/float(len(ref_token)+0.00000001))
            err += e
            cnt += len(ref_token)
        print(count, err,cnt,float(float(err+0.0) / float(cnt+0.01)*100))

compute_wer_txt(r"/teamscratch/tts_intern_experiment/yuwu1/librispeech_0.5"
                r"/results/spec_aug_ddp_cnn_mask15_speed_dev_other/data.json.rerank")
# def compute_wer(path,debug=False):
#     data = list(json.load(open(path, 'r'))['utts'].items())
#     err, cnt = 0, 0
#     for key, s in data:
#         tmp = []
#         for jj in range(len(s['output'])):
#             cur = s['output'][jj]
#             ref = cur['text'].split(' ')
#             for i in range(len(ref)):
#                 ref[i] = ref[i].replace("?","").replace("!","").replace(",","")
#             ref[-1] = ref[-1].replace(".","")
#             #print(s['rec_text'])
#             hyp = cur['rec_text'][1:-5].split(u'\u2581')
#             print(hyp)
#             #print(hyp)
#             e = editdistance.eval(ref, hyp)
#             tmp.append(e)
#
#         err += min(tmp)
#         cnt += len(ref)
#     print(err, cnt, float( float(err+0.0) / float(cnt+0.01) * 100))
#
# compute_wer(r"/teamscratch/tts_intern_experiment/yuwu1/librispeech_0.5"
#             r"/results/spec_aug_ddp_cnn_mask15_speed_test_other_68/data.json")
