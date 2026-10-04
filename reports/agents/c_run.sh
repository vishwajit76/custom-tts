tag=$1; shift
for c in punctuation questions commands conversational long_sentences edge_cases basic_hindi difficult_hindi; do
 .venv-v8/Scripts/python.exe scripts/gpu_lock.py .venv-v8/Scripts/python.exe evaluate.py --category $c --engine ${1:-goonj} | tail -1 | sed "s/^/$c /"
done > reports/agents/c_$tag.txt 2>&1
