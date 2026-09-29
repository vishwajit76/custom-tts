#!/usr/bin/env bash
# Export newest ckpt -> voices/hi_IN-custom-medium.onnx(+.json) and synthesise 3 test sentences.
set -euo pipefail
NAME=${NAME:-hi_f}; cd "$(dirname "$0")/.."; RUN=training/runs/$NAME
C=$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
STEP=$(python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['global_step'])" "$C")
mkdir -p voices "$RUN/samples/step_$STEP"
python -m piper.train.export_onnx --checkpoint "$C" --output-file voices/hi_IN-custom-medium.onnx
cp "$RUN/config.json" voices/hi_IN-custom-medium.onnx.json
i=0
for t in "नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।" "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ और संस्कृतियाँ एक साथ मिलकर रहती हैं।" "क्या आपने कल रात का खाना खा लिया? मुझे तो बहुत भूख लगी है!"; do
  i=$((i+1)); echo "$t" | python -m piper -m voices/hi_IN-custom-medium.onnx -f "$RUN/samples/step_$STEP/sample_$i.wav"
done
echo "exported step $STEP from $C; samples in $RUN/samples/step_$STEP"
