import os
p = r"\\gcrnfsw2-tts\tts_intern_experiment\yuwu1\30k_data\results\default_test_car\output.txt"
f = open(p,encoding="utf-8")
fw = open(p+".tsv","w",encoding="utf-8")
outputlocal = r"D:\user\yuwu\asr_result\car.output_35000.tsv"
outputdir = r"D:\user\yuwu\asr_result\car.output_35000"
fw2 = open(outputlocal,"w",encoding="utf-8")
for i, line in enumerate(f):
    if i < 35000:
        continue
    # if i >= 35000:
    #     break
    prediction, trans = line.strip().split("<eos>\t")

    prediction = prediction.replace("▁"," ").strip()
    #print(i, prediction, trans)

    fw2.write("{0}\t{1}\t{2}\n".format(i,trans,prediction))
fw2.close()
cmdline = r"D:\user\yuwu\compute_wer\CompMetricsMain.exe -l en-US -i {0} " \
          r"-idcol 0 -txcol 1 -recocol 2 -of {1}"
os.system(cmdline.format(outputlocal,outputdir))