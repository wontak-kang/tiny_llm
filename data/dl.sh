set -e
B=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for f in TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt; do
  echo "=== $f ==="
  curl -L -C - --retry 5 --retry-delay 3 -o "$f" "$B/$f"
done
echo "=== done ==="
ls -l
